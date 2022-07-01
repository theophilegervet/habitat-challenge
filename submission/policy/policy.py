from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import skimage.morphology
import math

import skimage.morphology
from .utils.morphology import binary_denoising, binary_dilation
from submission.utils.constants import MAX_DEPTH_REPLACEMENT_VALUE
from submission.utils.visualization_utils import draw_line


class Policy(nn.Module, ABC):
    """
    Policy to select high-level goals.
    """
    def __init__(self, config):
        super().__init__()
        self.hfov = config.ENVIRONMENT.hfov
        self.frame_width = config.ENVIRONMENT.frame_width

        self.denoise_goal_kernel = nn.Parameter(
            torch.from_numpy(
                skimage.morphology.disk(1)
            ).unsqueeze(0).unsqueeze(0).float(),
            requires_grad=False
        )
        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(
                skimage.morphology.disk(10)
            ).unsqueeze(0).unsqueeze(0).float(),
            requires_grad=False
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(
                skimage.morphology.disk(1)
            ).unsqueeze(0).unsqueeze(0).float(),
            requires_grad=False
        )

    @property
    @abstractmethod
    def goal_update_steps(self):
        pass

    def forward(self, map_features, local_pose, goal_category, obs):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 8 + num_sem_categories, M, M)
            local_pose: agent pose in local map
            goal_category: semantic goal category
            obs: frame containing (RGB, depth, segmentation) of shape
             (batch_size, 3 + 1 + num_sem_categories, frame_height, frame_width)

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size,)
            found_hint: binary variables to denote whether we found a hint of
             the object goal category of shape (batch_size,)
        """
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, goal_category)
        goal_map, found_hint = self.look_for_hint_in_frame(
            map_features, local_pose, goal_category, goal_map, found_goal, obs)
        goal_map = self.explore_otherwise(
            map_features, local_pose, goal_category, goal_map, found_goal, found_hint)
        return goal_map, found_goal, found_hint

    def look_for_hint_in_frame(self,
                               map_features,
                               local_pose,
                               goal_category,
                               goal_map,
                               found_goal,
                               obs):
        """
        If the goal category is not in the semantic map but is in the frame
        (beyond the maximum depth sensed and projected into the map) go
        towards it.
        """
        batch_size, _, map_size, _ = map_features.shape
        device = obs.device
        beyond_max_depth_mask = obs[:, 3, :, :] == MAX_DEPTH_REPLACEMENT_VALUE

        found_hint = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            # if not found_goal[e]:
            category_frame = obs[e, goal_category[e] + 4, :, :]

            # Only keep going if there's an instance of the goal
            # category in the frame detected beyond the maximum
            # depth sensed
            # if (category_frame[beyond_max_depth_mask[e]] == 1).sum() == 0:
            #     continue
            if (category_frame == 1).sum() == 0:
                continue

            # Select unexplored area
            frontier_map = (map_features[e, [1], :, :] == 0).float()

            # Dilate explored area
            frontier_map = 1 - binary_dilation(
                1 - frontier_map, self.dilate_explored_kernel)

            # Select the frontier
            frontier_map = binary_dilation(
                frontier_map, self.select_border_kernel) - frontier_map

            # Select the intersection between the frontier and the
            # direction of the object beyond the maximum depth sensed
            # TODO Refine the direction with the position of the object
            #  within the frame
            agent_angle = local_pose[e, 2].item()
            median_c = torch.nonzero(category_frame, as_tuple=True)[1].median()
            frame_angle = median_c / self.frame_width * self.hfov - self.hfov / 2
            print("agent_angle", agent_angle)
            print("mean_c", median_c)
            print("frame_angle", frame_angle)
            start_y = start_x = line_length = map_size // 2
            end_y = start_y + line_length * math.sin(math.radians(agent_angle))
            end_x = start_x + line_length * math.cos(math.radians(agent_angle))
            direction_map = torch.zeros(map_size, map_size)
            draw_line((start_y, start_x), (end_y, end_x), direction_map, steps=line_length)
            direction_map = direction_map.to(frontier_map.device)
            goal_map[e] = frontier_map.squeeze(0) * direction_map
            found_hint[e] = True

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
                          local_pose,
                          goal_category,
                          goal_map,
                          found_goal,
                          found_hint):
        """
        If the goal category is neither in the semantic map nor in the frame,
        explore with the child policy.
        """
        pass
