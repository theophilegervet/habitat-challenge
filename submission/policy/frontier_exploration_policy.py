from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import skimage.morphology

from .policy import Policy


def binary_dilation(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return torch.clamp(
        torch.nn.functional.conv2d(
            binary_image,
            kernel,
            padding=kernel.shape[-1] // 2
        ),
        0, 1
    )


def binary_erosion(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return 1 - torch.clamp(
        torch.nn.functional.conv2d(
            1 - binary_image,
            kernel,
            padding=kernel.shape[-1] // 2
        ),
        0, 1
    )


def binary_opening(binary_image, kernel):
    return binary_dilation(binary_erosion(binary_image, kernel), kernel)


def binary_closing(binary_image, kernel):
    return binary_erosion(binary_dilation(binary_image, kernel), kernel)


def binary_denoising(binary_image, kernel):
    return binary_opening(binary_closing(binary_image, kernel), kernel)


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
        self.denoise_goal_kernel = nn.Parameter(
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
        goal_category_cpu = goal_category.cpu().numpy()

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

            # If we think we've found the goal, let's add a bit of extra
            # engineering to remove false positives:
            if (category_map == 1).sum() > 0:
                if goal_category_cpu[e] == 1:
                    # If we're looking for a couch, filter out all cells that
                    # also have been classified as a chair or a bed
                    category_map -= map_features[e, 0 + 8, :, :]
                    category_map -= map_features[e, 3 + 8, :, :]

                elif goal_category_cpu[e] == 3:
                    # If we're looking for a bed, filter out couch
                    category_map -= map_features[e, 1 + 8, :, :]

                elif goal_category_cpu[e] == 0:
                    # If we're looking for a chair, filter out couch and
                    # bed (frame)
                    category_map -= map_features[e, 1 + 8, :, :]
                    category_map -= map_features[e, 3 + 8, :, :]

                # Remove noise with standard morphological transformation
                # (closing -> opening)
                # TODO Commenting this out to check whether it caused a regression
                # category_map = binary_denoising(
                #     category_map.unsqueeze(0).unsqueeze(0),
                #     self.denoise_goal_kernel
                # ).squeeze(0).squeeze(0)

            if (category_map == 1).sum() > 0:
                goal_map[e] = category_map == 1
                found_goal[e] = True

            # Else, set unexplored area as the goal
            else:
                goal_map[e] = frontier_map[e]

        return goal_map, found_goal, regression_logits
