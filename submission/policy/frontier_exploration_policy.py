from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import skimage.morphology

from .policy import Policy


def binary_dilation(binary_image, kernel):
    return torch.clamp(
        torch.nn.functional.conv2d(
            binary_image,
            kernel,
            padding=kernel.shape[-1] // 2
        ),
        0, 1
    )


class FrontierExplorationPolicy(Policy):
    """
    This policy picks the closest non-explored point.
    """

    def __init__(self):
        super().__init__()

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

    def forward(self,
                map_features: Tensor,
                goal_category: Tensor
                ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        batch_size, _, height, width = map_features.shape
        device = map_features.device

        # Select unexplored area
        frontier_map = (map_features[:, [1], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel)

        # Select the frontier
        frontier_map = binary_dilation(
            frontier_map, self.select_border_kernel) - frontier_map

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal = torch.zeros(batch_size, dtype=torch.bool, device=device)
        regression_logits = None

        for e in range(batch_size):
            # If the object goal category is present in the local map, go to it
            category_map = map_features[e, goal_category[e] + 8, :, :]
            if (category_map == 1).sum() > 0:
                goal_map[e] = category_map == 1
                found_goal[e] = True

            # Else, set unexplored area as the goal
            else:
                goal_map[e] = frontier_map[e]

        return goal_map, found_goal, regression_logits
