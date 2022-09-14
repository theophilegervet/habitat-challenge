import json
import os
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from scipy.special import expit
import skimage.morphology

from habitat import Config, make_dataset
from habitat.core.env import RLEnv
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.dataset import ALL_SCENES_MASK
from gym.spaces import Dict as SpaceDict
from gym.spaces import Box, Discrete
from ray.rllib.env.env_context import EnvContext

from submission.obs_preprocessor.obs_preprocessor import ObsPreprocessor
from submission.planner.planner import Planner
from submission.planner.fmm_planner import FMMPlanner
from submission.visualizer.visualizer import Visualizer
from submission.semantic_map.semantic_map_state import SemanticMapState
from submission.semantic_map.semantic_map_module import SemanticMapModule
from submission.policy.frontier_exploration_policy import FrontierExplorationPolicy
# Import to register dataset in environment processes
from submission.dataset.semexp_policy_training_dataset import SemanticExplorationPolicyTrainingDataset


class SemanticExplorationPolicyTrainingEnvWrapper(RLEnv):
    """
    This environment wrapper is used to train the semantic exploration
    policy with reinforcement learning. It contains stepping the underlying
    environment, preprocessing observations, storing and updating the
    semantic map state, planning given a high-level goal predicted by
    the policy, and computing rewards. It is complemented by the high-level
    goal policy to be trained.
    """

    def __init__(self,
                 rllib_config: Optional[EnvContext] = None,
                 config: Optional[Config] = None
                 ):
        assert rllib_config is not None or config is not None

        if config is None:
            config = rllib_config["config"]
            config.defrost()

            # Select scenes
            if config.TASK_CONFIG.DATASET.TYPE == "SemexpPolicyTraining":
                dataset = SemanticExplorationPolicyTrainingDataset(
                    config.TASK_CONFIG.DATASET)
                scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
                if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
                    scenes = [dataset.scene_from_scene_path(scene_id)
                              for scene_id in dataset.scene_ids]
            else:
                dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
                scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
                if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
                    scenes = dataset.get_scenes_to_load(
                        config.TASK_CONFIG.DATASET)
            del dataset
            if len(scenes) == 1:
                # Same scene for all workers
                scene_splits = [[scenes[0]] for _ in range(rllib_config.num_workers)]
            elif len(scenes) >= rllib_config.num_workers:
                # Scenes distributed among workers
                scene_splits = [[] for _ in range(rllib_config.num_workers)]
                for idx, scene in enumerate(scenes):
                    scene_splits[idx % len(scene_splits)].append(scene)
                assert sum(map(len, scene_splits)) == len(scenes)
            else:  # 1 < len(scenes) < rllib_config.num_workers
                # Workers distributed among scenes
                scene_splits = [[scenes[idx % len(scenes)]]
                                for idx in range(rllib_config.num_workers)]
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = scene_splits[
                rllib_config.worker_index - 1]

            # Set random seed
            config.TASK_CONFIG.SEED = rllib_config.worker_index

            config.freeze()

        super().__init__(config=config.TASK_CONFIG)

        assert config.NUM_ENVIRONMENTS == 1
        self.dataset_type = config.TASK_CONFIG.DATASET.TYPE
        if self.dataset_type == "SemexpPolicyTraining":
            if "unannotated_scenes" in config.TASK_CONFIG.DATASET.DATA_PATH:
                self.ground_truth_semantic_map_type = "predicted_first_person"
            else:
                self.ground_truth_semantic_map_type = "annotations_top_down"
            self.dense_goal_rew_coeff = config.TRAIN.RL.dense_goal_rew_coeff
        self.device = (torch.device("cpu") if config.NO_GPU else
                       torch.device(f"cuda:{self.habitat_env.sim.gpu_device}"))
        self.goal_update_steps = config.AGENT.POLICY.SEMANTIC.goal_update_steps
        self.max_steps = 500 // self.goal_update_steps
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0
        self.intrinsic_rew_coeff = config.TRAIN.RL.intrinsic_rew_coeff
        self.gamma = config.TRAIN.RL.gamma
        self.inference_downscaling = config.AGENT.POLICY.SEMANTIC.inference_downscaling
        self.print_images = config.PRINT_IMAGES

        self.planner = Planner(config)
        self.visualizer = Visualizer(config)
        self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)
        self.semantic_map = SemanticMapState(config, self.device)
        self.semantic_map_module = SemanticMapModule(config).to(self.device)
        # We only use methods of the abstract base class
        self.policy = FrontierExplorationPolicy(config).to(self.device)

        self.observation_space = SpaceDict({
            "map_features": Box(
                low=0.,
                high=1.,
                shape=(
                    8 + self.semantic_map.num_sem_categories,
                    self.semantic_map.local_h // self.inference_downscaling,
                    self.semantic_map.local_w // self.inference_downscaling
                ),
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

        # Action space is logit pre sigmoid normalization to [0, 1]
        self.action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )

        self.scene_id = None
        self.episode_id = None
        self.timestep = None
        self.goal_category_tensor = None
        self.goal_category = None
        self.goal_name = None
        self.infos = None

        if self.dataset_type == "SemexpPolicyTraining":
            self.gt_planner = None
            self.gt_map_resolution = None
            self.xz_origin_cm = None

    def reset(self) -> dict:
        self.timestep = 0
        self.infos = []

        self.obs_preprocessor.reset()
        self.planner.reset()
        self.semantic_map.init_map_and_pose()

        obs = super().reset()
        seq_obs = [obs]

        scene_dir = "/".join(self.current_episode.scene_id.split("/")[:-1])
        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id

        if self.print_images:
            self.visualizer.reset()
            self.visualizer.set_vis_dir(self.scene_id, self.episode_id)

        for _ in range(self.panorama_start_steps):
            obs, _, _, _ = super().step(HabitatSimActions.TURN_RIGHT)
            seq_obs.append(obs)

        (
            map_features,
            semantic_frame,
            local_pose,
            self.goal_category_tensor,
            self.goal_name
        ) = self._update_map(seq_obs, update_global=True)
        self.goal_category = self.goal_category_tensor.item()

        # If we are training on a dataset generated specifically for
        # semantic exploration policy training, create the ground-truth
        # planner that allows us to compute dense distance to goal reward
        if self.dataset_type == "SemexpPolicyTraining":
            map_dir = (
                scene_dir +
                f"/floor_semantic_maps_{self.ground_truth_semantic_map_type}"
            )
            with open(f"{map_dir}/{self.scene_id}_info.json", "r") as f:
                scene_info = json.load(f)
            start_height_cm = self.current_episode.start_position[1] * 100.
            floor_heights_cm = scene_info["floor_heights_cm"]
            self.xz_origin_cm = scene_info["xz_origin_cm"]
            self.gt_map_resolution = scene_info["map_generation_parameters"][
                "resolution_cm"]
            floor_idx = min(
                range(len(floor_heights_cm)),
                key=lambda idx: abs(floor_heights_cm[idx] - start_height_cm)
            )
            sem_map = np.load(f"{map_dir}/{self.scene_id}_floor{floor_idx}.npy")
            selem = skimage.morphology.disk(2)
            traversible = skimage.morphology.binary_dilation(
                sem_map[0], selem) != True
            traversible = 1 - traversible
            self.gt_planner = FMMPlanner(traversible)
            selem = skimage.morphology.disk(int(100 / self.gt_map_resolution))
            goal_map = skimage.morphology.binary_dilation(
                sem_map[self.goal_category + 1], selem) != True
            goal_map = 1 - goal_map
            self.gt_planner.set_multi_goal(goal_map)

        if self.print_images:
            vis_inputs = {
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(0),
                "obstacle_map": self.semantic_map.get_obstacle_map(0),
                "goal_map": self.semantic_map.get_goal_map(0),
                "found_goal": False,
                "found_hint": False,
                "explored_map": self.semantic_map.get_explored_map(0),
                "semantic_map": self.semantic_map.get_semantic_map(0),
                "semantic_frame": semantic_frame,
                "goal_name": self.goal_name,
                "goal_category": self.goal_category,
                "timestep": self.timestep
            }
            self.visualizer.visualize(**vis_inputs)

        map_features = F.avg_pool2d(map_features, self.inference_downscaling)
        obs = {
            "map_features": map_features[0].cpu().numpy(),
            "local_pose": local_pose[0].cpu().numpy(),
            "goal_category": self.goal_category
        }
        return obs

    def _compute_distance_to_goal(self, position: List[float]) -> float:
        map_x = int((position[0] * 100. - self.xz_origin_cm[0]) / self.gt_map_resolution)
        map_z = int((position[2] * 100. - self.xz_origin_cm[1]) / self.gt_map_resolution)
        return self.gt_planner.fmm_dist[map_x, map_z]

    def step(self, goal_action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        self.timestep += 1
        prev_explored_area = self.semantic_map.global_map[0, 1].sum()
        if self.dataset_type == "SemexpPolicyTraining":
            prev_distance_to_goal = self._compute_distance_to_goal(
                list(self.habitat_env.sim.get_agent_state().position))

        # Set high-level goal predicted by the policy
        goal_location = (expit(goal_action) * (self.semantic_map.local_h - 1)).astype(int)
        goal_map = np.zeros((
            self.semantic_map.local_h,
            self.semantic_map.local_w
        ))
        goal_map[goal_location[0], goal_location[1]] = 1
        self.semantic_map.update_global_goal_for_env(0, goal_map)

        # For each low-level step
        for t in range(self.goal_update_steps):
            # 1 - Plan
            planner_inputs = {
                "obstacle_map": self.semantic_map.get_obstacle_map(0),
                "goal_map": self.semantic_map.get_goal_map(0),
                "found_goal": False,
                "found_hint": False,
                "goal_category": self.goal_category,
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(0)
            }
            action = self.planner.plan(**planner_inputs)

            # 2 - Step
            obs, _, _, _ = super().step(action)

            # 3 - Update map
            map_features, semantic_frame, local_pose, _, _ = self._update_map(
                [obs],
                update_global=(t == self.goal_update_steps - 1)
            )

            # 4 - Check whether we found the goal
            if self.dataset_type == "SemexpPolicyTraining":
                curr_distance_to_goal = self._compute_distance_to_goal(
                    list(self.habitat_env.sim.get_agent_state().position))
                found_goal = (curr_distance_to_goal < 10.0)
            else:
                _, found_goal = self.policy.reach_goal_if_in_map(
                    map_features,
                    self.goal_category_tensor
                )
                found_goal = found_goal.item()

            if found_goal:
                break

        if self.print_images:
            # Visualize the final state
            vis_inputs = {
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(0),
                "obstacle_map": self.semantic_map.get_obstacle_map(0),
                "goal_map": self.semantic_map.get_goal_map(0),
                "found_goal": False,
                "found_hint": False,
                "explored_map": self.semantic_map.get_explored_map(0),
                "semantic_map": self.semantic_map.get_semantic_map(0),
                "semantic_frame": semantic_frame,
                "goal_name": self.goal_name,
                "goal_category": self.goal_category,
                "timestep": self.timestep
            }
            self.visualizer.visualize(**vis_inputs)

        map_features = F.avg_pool2d(map_features, self.inference_downscaling)
        obs = {
            "map_features": map_features[0].cpu().numpy(),
            "local_pose": local_pose[0].cpu().numpy(),
            "goal_category": self.goal_category
        }

        # Intrinsic reward = increase in explored area (in m^2)
        # Sparse goal reward = binary found goal or not
        # Dense goal reward = decrease in geodesic distance to goal (in map cells)

        curr_explored_area = self.semantic_map.global_map[0, 1].sum()
        intrinsic_reward = (curr_explored_area - prev_explored_area).item()
        intrinsic_reward *= (self.semantic_map.resolution / 100) ** 2

        sparse_goal_reward = 1. if (found_goal and self.timestep > 1) else 0.

        if self.dataset_type == "SemexpPolicyTraining":
            dense_goal_reward = prev_distance_to_goal - curr_distance_to_goal

            reward = (dense_goal_reward * self.dense_goal_rew_coeff +
                      intrinsic_reward * self.intrinsic_rew_coeff)
        else:
            reward = sparse_goal_reward + intrinsic_reward * self.intrinsic_rew_coeff

        info = {
            "timestep": self.timestep,
            "sparse_goal_reward": sparse_goal_reward,
            "intrinsic_reward": intrinsic_reward * self.intrinsic_rew_coeff,
            "unscaled_intrinsic_reward": intrinsic_reward,
            "discounted_sparse_goal_reward": sparse_goal_reward * (self.gamma ** (self.timestep - 1)),
            "discounted_unscaled_intrinsic_reward": intrinsic_reward * (self.gamma ** (self.timestep - 1)),
            "action_0": float(goal_action[0]),
            "action_1": float(goal_action[1]),
        }
        if self.dataset_type == "SemexpPolicyTraining":
            info["curr_distance_to_goal"] = curr_distance_to_goal
            info["dense_goal_reward"] = dense_goal_reward * self.dense_goal_rew_coeff
            info["unscaled_dense_goal_reward"] = dense_goal_reward
            info["discounted_unscaled_dense_goal_reward"] = dense_goal_reward * (self.gamma ** (self.timestep - 1))

        self.infos.append(info)

        done = found_goal or self.timestep == self.max_steps - 1
        if done and self.print_images:
            json.dump(
                self.infos,
                open(os.path.join(self.visualizer.vis_dir, "infos.json"), "w"),
                indent=4
            )

        return obs, reward, done, info

    def _update_map(self,
                    seq_obs: List[Observations],
                    update_global: bool,
                    ) -> Tuple[Tensor, np.ndarray, Tensor, Tensor, str]:
        """Update the semantic map with a sequence of observations.

        Arguments:
            seq_obs: sequence of observations
            update_global: if True, update the global map and pose at the
             last timestep

        Returns:
            final_map_features: final semantic map features of shape
             (1, 8 + num_sem_categories, M, M)
            final_semantic_frame: final semantic frame visualization
            final_local_pose: final local pose of shape (1, 3)
            goal_category: semantic goal category ID
            goal_name: semantic goal category
        """
        # Preprocess observations
        sequence_length = len(seq_obs)
        (
            seq_obs_preprocessed,
            seq_semantic_frame,
            seq_pose_delta,
            goal_category,
            goal_name
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

        return (
            seq_map_features[:, -1],
            seq_semantic_frame[-1],
            seq_local_pose[:, -1],
            goal_category,
            goal_name
        )

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
