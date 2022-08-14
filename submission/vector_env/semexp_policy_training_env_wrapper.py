import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List

from habitat import Config
from habitat.core.env import RLEnv
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from gym.spaces import Dict as SpaceDict
from gym.spaces import Box, Discrete

from submission.obs_preprocessor.obs_preprocessor import ObsPreprocessor
from submission.planner.planner import Planner
from submission.visualizer.visualizer import Visualizer
from submission.semantic_map.semantic_map_state import SemanticMapState
from submission.semantic_map.semantic_map_module import SemanticMapModule
from submission.policy.frontier_exploration_policy import FrontierExplorationPolicy


class SemanticExplorationPolicyTrainingEnvWrapper(RLEnv):
    """
    This environment wrapper is used to train the semantic exploration
    policy with reinforcement learning. It contains stepping the underlying
    environment, preprocessing observations, storing and updating the
    semantic map state, planning given a high-level goal predicted by
    the policy, and computing rewards. It is complemented by the high-level
    goal policy to be trained.
    """

    def __init__(self, config: Config):
        super().__init__(config=config.TASK_CONFIG)

        assert config.NUM_ENVIRONMENTS == 1
        self.device = (torch.device("cpu") if config.NO_GPU else
                       torch.device(f"cuda:{self.habitat_env.sim.gpu_device}"))
        self.max_steps = config.AGENT.max_steps
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0
        self.goal_update_steps = config.AGENT.POLICY.SEMANTIC.goal_update_steps
        self.intrinsic_rew_coeff = config.TRAIN.RL.intrinsic_rew_coeff

        self.planner = Planner(config)
        self.visualizer = Visualizer(config)
        self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)
        self.semantic_map = SemanticMapState(config, self.device)
        self.semantic_map_module = SemanticMapModule(config).to(self.device)
        # We only use methods of the abstract base class
        self.policy = FrontierExplorationPolicy(config).to(self.device)

        self.observation_space = SpaceDict({
            "map_features": Box(
                low=0,
                high=1,
                shape=(self.semantic_map.local_h, self.semantic_map.local_w),
                dtype=np.float32
            ),
            "local_pose": Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            "goal_category": Discrete(6),
        })

        self.action_space = Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

        self.scene_id = None
        self.episode_id = None
        self.timestep = None
        self.goal_category = None

    def reset(self) -> dict:
        self.obs_preprocessor.reset()
        self.planner.reset()
        self.semantic_map.init_map_and_pose()

        obs = super().reset()
        seq_obs = [obs]
        self.timestep = 0

        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id

        for _ in range(self.panorama_start_steps):
            obs, _, _, _ = super().step(HabitatSimActions.TURN_RIGHT)
            seq_obs.append(obs)
            self.timestep += 1

        (
            map_features,
            local_pose,
            self.goal_category
        ) = self._update_map(seq_obs, update_global=True)

        obs = {
            "map_features": map_features,
            "local_pose": local_pose,
            "goal_category": self.goal_category
        }
        return obs

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        prev_explored_area = self.semantic_map.global_map[0, 1].sum()

        # 1 - Set high-level goal predicted by the policy
        goal_location = (action * (self.semantic_map.local_h - 1)).astype(int)
        goal_map = np.zeros((
            self.semantic_map.local_h,
            self.semantic_map.local_w
        ))
        goal_map[goal_location[0], goal_location[1]] = 1
        self.semantic_map.update_global_goal_for_env(0, goal_map)

        for t in range(self.goal_update_steps):
            # 2 - Plan
            planner_inputs = {
                "obstacle_map": self.semantic_map.get_obstacle_map(0),
                "goal_map": self.semantic_map.get_goal_map(0),
                "found_goal": False,
                "found_hint": False,
                "goal_category": self.goal_category,
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(0)
            }
            action = self.planner.plan(**planner_inputs)

            # 3 - Step
            obs, _, _, _ = super().step(action)

            # 4 - Update map
            map_features, local_pose, _ = self._update_map(
                [obs],
                update_global=(t == self.goal_update_steps - 1)
            )

            # 5 - Check whether we found the goal
            _, found_goal = self.policy.reach_goal_if_in_map(
                map_features,
                self.goal_category
            )
            found_goal = found_goal.item()

            if found_goal or self.timestep > self.max_steps:
                break

        obs = {
            "map_features": map_features,
            "local_pose": local_pose,
            "goal_category": self.goal_category
        }

        # Intrinsic reward = increase in explored area (in m^2)
        curr_explored_area = self.semantic_map.global_map[0, 1].sum()
        intrinsic_reward = (curr_explored_area - prev_explored_area)
        intrinsic_reward *= (self.semantic_map.resolution / 100) ** 2

        if found_goal:
            goal_reward = 1.
            done = True
        elif self.timestep > self.max_steps:
            goal_reward = 0.
            done = True
        else:
            goal_reward = 0.
            done = False

        reward = goal_reward + intrinsic_reward * self.intrinsic_rew_coeff

        return obs, reward, done, {}

    def _update_map(self,
                    seq_obs: List[Observations],
                    update_global: bool,
                    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Update the semantic map with a sequence of observations.

        Arguments:
            seq_obs: sequence of observations
            update_global: if True, update the global map and pose at the
             last timestep

        Returns:
            final_map_features: final semantic map features of shape
             (1, 8 + num_sem_categories, M, M)
            final_local_pose: final local pose of shape (1, 3)
            goal_category: semantic goal category
        """
        # Preprocess observations
        sequence_length = len(seq_obs)
        (
            seq_obs_preprocessed,
            seq_pose_delta,
            goal_category,
        ) = self.obs_preprocessor.preprocess_sequence(seq_obs)

        seq_dones = torch.tensor([False] * sequence_length)
        seq_update_global = torch.tensor([False] * sequence_length)
        seq_update_global[-1] = update_global

        # Update map with observations and generate map features
        (
            seq_map_features,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs_preprocessed.unsqueeze(0),
            seq_pose_delta.unsqueeze(0).to(self.device),
            seq_dones.unsqueeze(0).to(self.device),
            seq_update_global.unsqueeze(0).to(self.device),
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

        return seq_map_features[:, -1], seq_local_pose[:, -1], goal_category

    def get_reward_range(self):
        """Required by RLEnv but not used."""
        pass

    def get_reward(self, observations: Observations):
        """Required by RLEnv but not used."""
        pass

    def get_done(self, observations: Observations):
        """Required by RLEnv but not used."""
        pass

    def get_info(self, observations: Observations):
        """Required by RLEnv but not used."""
        pass
