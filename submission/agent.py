from typing import List, Tuple, Dict
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel
import time

import habitat
from habitat import Config
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from .agent_module import AgentModule
from .semantic_map.semantic_map_state import SemanticMapState
from .planner.planner import Planner
from .visualizer.visualizer import Visualizer
from .obs_preprocessor.obs_preprocessor import ObsPreprocessor
from .policy.frontier_exploration_policy import FrontierExplorationPolicy
from .policy.semantic_exploration_policy import SemanticExplorationPolicy


exploration_policies = {
    "frontier": FrontierExplorationPolicy,
    "semantic": SemanticExplorationPolicy
}


class Agent(habitat.Agent):
    """
    This class is the agent that acts in the environment.
    """

    def __init__(self, config: Config, rank: int, ddp: bool = False):
        self.max_steps = config.AGENT.max_steps
        self.precision = torch.float16 if config.MIXED_PRECISION else torch.float32
        self.num_environments = config.NUM_ENVIRONMENTS
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        policy = exploration_policies[config.AGENT.POLICY.type](config)
        self.module = AgentModule(config, policy)

        if config.NO_GPU:
            self.device = torch.device("cpu")

        else:
            self.device_id = config.AGENT_GPU_IDS[rank]
            self.device = torch.device(f"cuda:{self.device_id}")
            self.module = self.module.to(self.device)
            if ddp:
                self.module = DistributedDataParallel(
                    self.module, device_ids=[self.device_id])
            else:
                # Use DataParallel only as a wrapper to move model inputs to GPU
                self.module = DataParallel(
                    self.module, device_ids=[self.device_id])

        self.obs_preprocessor = ObsPreprocessor(
            config, self.num_environments, self.device)
        self.semantic_map = SemanticMapState(config, rank)
        self.planner = Planner(config)
        self.visualizer = Visualizer(config)

        self.timesteps = None

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(self,
                               obs: torch.Tensor,
                               pose_delta: torch.Tensor,
                               goal_category: torch.Tensor
                               ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
        the main inference function of the agent that lets it interact with
        vectorized environments.

        This function assumes that the agent has been initialized.

        Args:
            obs: current frame containing (RGB, depth, segmentation) of shape
             (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            goal_category: semantic goal category

        Returns:
            planner_inputs: list of num_environments planner inputs dicts containing
                obstacle_map: (M, M) binary np.ndarray local obstacle map
                 prediction
                sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                 and local map boundaries planning window (gx1, gx2, gy1, gy2)
                goal_map: (M, M) binary np.ndarray denoting goal location
            vis_inputs: list of num_environments visualization info dicts containing
                explored_map: (M, M) binary np.ndarray local explored map
                 prediction
                semantic_map: (M, M) np.ndarray containing local semantic map
                 predictions
        """
        dones = torch.tensor([False] * self.num_environments)
        # TODO update_global should be computed in self.module instead of
        #  accessing self.module.module.policy here
        # update_global = torch.tensor(
        #     [self.timesteps[e] % self.module.module.policy.goal_update_steps == 0
        #      for e in range(self.num_environments)])
        # TODO Temp to debug semantic policy on CPU
        update_global = torch.tensor(
            [self.timesteps[
                 e] % self.module.policy.goal_update_steps == 0
             for e in range(self.num_environments)])
        self.timesteps = [self.timesteps[e] + 1
                          for e in range(self.num_environments)]

        (
            goal_map,
            found_goal,
            found_hint,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.to(self.precision).unsqueeze(1),
            pose_delta.unsqueeze(1),
            goal_category.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = goal_map.squeeze(1).cpu().numpy()
        found_goal = found_goal.squeeze(1).cpu()
        goal_category = goal_category.cpu()

        for e in range(self.num_environments):
            if found_goal[e] or found_hint[e] or update_global[e]:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "found_goal": found_goal[e].item(),
                "goal_category": goal_category[e].item(),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e)
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.semantic_map.get_explored_map(e),
                "semantic_map": self.semantic_map.get_semantic_map(e),
                "timestep": self.timesteps[e]
            }
            for e in range(self.num_environments)
        ]

        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.semantic_map.init_map_and_pose()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.semantic_map.init_map_and_pose_for_env(e)

    # ------------------------------------------------------------------
    # Inference methods to interact with a single un-vectorized environment
    # ------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.obs_preprocessor.reset()
        self.planner.reset()
        self.visualizer.reset()

    @torch.no_grad()
    def act(self, obs: Observations) -> Dict[str, int]:
        """Act end-to-end."""
        # t0 = time.time()

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            semantic_frame,
            pose_delta,
            goal_category,
            goal_name
        ) = self.obs_preprocessor.preprocess([obs])

        # t1 = time.time()
        # print(f"[Agent] Obs preprocessing time: {t1 - t0:.2f}")

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed, pose_delta, goal_category)

        # t2 = time.time()
        # print(f"[Agent] Semantic mapping and policy time: {t2 - t1:.2f}")

        # 3 - Planning
        if self.timesteps[0] < self.panorama_start_steps:
            action = HabitatSimActions.TURN_RIGHT
        elif self.timesteps[0] > self.max_steps:
            action = HabitatSimActions.STOP
        else:
            action = self.planner.plan(**planner_inputs[0])
        self.obs_preprocessor.last_actions[0] = action

        # t3 = time.time()
        # print(f"[Agent] Planning time: {t3 - t2:.2f}")

        # 4 - Visualization
        vis_inputs[0]["semantic_frame"] = semantic_frame[0]
        vis_inputs[0]["goal_name"] = goal_name[0]
        self.visualizer.visualize(**planner_inputs[0], **vis_inputs[0])

        # t4 = time.time()
        # print(f"[Agent] Visualization time: {t4 - t3:.2f}")
        # print(f"[Agent] Total time: {t4 - t0:.2f}")
        # print()

        return {"action": action}

    def set_vis_dir(self, scene_id: str, episode_id: str):
        """
        Reset visualization directory - if we have access to
        environment object and scene_id and episode_id.
        """
        self.planner.set_vis_dir(scene_id, episode_id)
        self.visualizer.set_vis_dir(scene_id, episode_id)
