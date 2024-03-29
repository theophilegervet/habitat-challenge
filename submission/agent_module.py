import torch.nn as nn
import time

from .semantic_map.semantic_map_module import SemanticMapModule
from .policy.policy import Policy


class AgentModule(nn.Module):

    def __init__(self, config, policy: Policy):
        super().__init__()

        self.semantic_map_module = SemanticMapModule(config)
        self.policy = policy

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

    def forward(self,
                seq_obs,
                seq_pose_delta,
                seq_goal_category,
                seq_dones,
                seq_update_global,
                init_local_map,
                init_global_map,
                init_local_pose,
                init_global_pose,
                init_lmb,
                init_origins):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_goal_category: sequence of goal categories of shape
             (batch_size, sequence_length, 1)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            seq_found_hint: binary variables to denote whether we found a hint of
             the object goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # t0 = time.time()

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        # Predict high-level goals from map features
        # batched across sequence length x num environments
        map_features = seq_map_features.flatten(0, 1)
        local_pose = seq_local_pose.flatten(0, 1)
        goal_category = seq_goal_category.flatten(0, 1)
        obs = seq_obs.flatten(0, 1)
        (
            goal_map,
            found_goal,
            found_hint
        ) = self.policy(map_features, local_pose, goal_category, obs)
        seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])
        seq_found_goal = found_goal.view(batch_size, sequence_length)
        seq_found_hint = found_hint.view(batch_size, sequence_length)

        # t2 = time.time()
        # print(f"[Policy] Total time: {t2 - t1:.2f}")

        return (
            seq_goal_map,
            seq_found_goal,
            seq_found_hint,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )
