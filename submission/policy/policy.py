from abc import ABC, abstractmethod
import torch
import torch.nn as nn

import skimage.morphology
from .utils.morphology import binary_denoising
from submission.utils.constants import MAX_DEPTH_REPLACEMENT_VALUE


class Policy(nn.Module, ABC):
    """
    Policy to select high-level goals.
    """
    def __init__(self, config):
        super().__init__()

        self.denoise_goal_kernel = nn.Parameter(
            torch.from_numpy(
                skimage.morphology.disk(1)
            ).unsqueeze(0).unsqueeze(0).float(),
            requires_grad=False
        )

    @property
    @abstractmethod
    def goal_update_steps(self):
        pass

    def forward(self, map_features, global_pose, goal_category, obs):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 8 + num_sem_categories, M, M)
            global_pose: global agent pose
            goal_category: semantic goal category
            obs: frame containing (RGB, depth, segmentation) of shape
             (batch_size, 3 + 1 + num_sem_categories, frame_height, frame_width)

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size,)
        """
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, goal_category)
        goal_map, found_hint = self.look_for_hint_in_frame(
            obs, global_pose, goal_category, goal_map, found_goal)
        goal_map = self.explore_otherwise(
            map_features, global_pose, goal_category, goal_map, found_goal, found_hint)
        return goal_map, found_goal

    def look_for_hint_in_frame(self, obs, global_pose, goal_category, goal_map, found_goal):
        """
        If the goal category is not in the semantic map but is in the frame
        (beyond the maximum depth sensed and projected into the map) go
        towards it.
        """
        batch_size = obs.shape[0]
        device = obs.device
        beyond_max_depth_mask = obs[:, 3, :, :] == MAX_DEPTH_REPLACEMENT_VALUE

        found_hint = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            if not found_goal[e]:
                category_frame = obs[e, goal_category[e] + 4, :, :]

                if (category_frame == 1).sum() > 0:
                    print("Object in frame!")

                if (category_frame[beyond_max_depth_mask[e]] == 1).sum() > 0:
                    print("Object in frame beyond max depth!")

        return goal_map, found_hint

    def reach_goal_if_in_map(self, map_features, goal_category):
        """If the goal category is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        goal_category_cpu = goal_category.cpu().numpy()

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            category_map = map_features[e, goal_category[e] + 8, :, :]

            # If we think we've found the goal, let's add a bit of extra
            # engineering to filter out false positives:
            if (category_map == 1).sum() > 0:
                if goal_category_cpu[e] == 1:
                    # If we're looking for a couch, filter out all cells that
                    # also have been classified as a chair or a bed
                    category_map -= (map_features[e, 0 + 8, :, :] == 1).float()
                    category_map -= (map_features[e, 3 + 8, :, :] == 1).float()

                elif goal_category_cpu[e] == 3:
                    # If we're looking for a bed, filter out couch
                    category_map -= (map_features[e, 1 + 8, :, :] == 1).float()

                elif goal_category_cpu[e] == 0:
                    # If we're looking for a chair, filter out couch and
                    # bed (frame)
                    category_map -= (map_features[e, 1 + 8, :, :] == 1).float()
                    category_map -= (map_features[e, 3 + 8, :, :] == 1).float()

                if goal_category_cpu[e] in [0, 1, 3]:
                    # For large objects (chair, couch, bed), remove noise with
                    # standard morphological transformation (closing -> opening)
                    category_map = binary_denoising(
                        category_map.unsqueeze(0).unsqueeze(0),
                        self.denoise_goal_kernel
                    ).squeeze(0).squeeze(0)

            if (category_map == 1).sum() > 0:
                goal_map[e] = category_map == 1
                found_goal[e] = True

        # TODO If the goal is in the semantic map below the category prediction
        #  threshold, try going towards it to check whether the detection is
        #  correct - this could backfire and introduce false positives

        return goal_map, found_goal

    @abstractmethod
    def explore_otherwise(self,
                          map_features,
                          global_pose,
                          goal_category,
                          goal_map,
                          found_goal,
                          found_hint):
        """
        If the goal category is neither in the semantic map nor in the frame,
        explore with the child policy.
        """
        pass
