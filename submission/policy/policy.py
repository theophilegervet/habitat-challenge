from abc import ABC, abstractmethod
import torch
import torch.nn as nn

import skimage.morphology
from .utils.morphology import binary_denoising


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

    def forward(self, map_features, global_pose, goal_category):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, channels, M, M)
            global_pose: global agent pose
            goal_category: semantic goal category

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size,)
        """
        goal_map, found_goal = self.reach_goal_if_found(map_features, goal_category)
        goal_map = self.explore_otherwise(
            map_features, global_pose, goal_category, goal_map, found_goal)
        return goal_map, found_goal

    def reach_goal_if_found(self, map_features, goal_category):
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

        return goal_map, found_goal

    @abstractmethod
    def explore_otherwise(self,
                          map_features,
                          global_pose,
                          goal_category,
                          goal_map,
                          found_goal):
        pass
