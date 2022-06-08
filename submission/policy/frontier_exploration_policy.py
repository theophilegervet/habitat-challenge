from typing import Tuple, Optional
import torch
from torch import Tensor
import skimage.morphology
import numpy as np

from .policy import Policy


class FrontierExplorationPolicy(Policy):
    """
    This policy picks the closest non-explored point.
    """

    def forward(self,
                map_features: Tensor,
                goal_category: Tensor
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO Move this to GPU
        batch_size, _, height, width = map_features.shape
        map_features = map_features.cpu().numpy()
        goal_category = goal_category.cpu().numpy()

        goal_map = np.zeros((batch_size, height, width))

        for e in range(batch_size):
            # If the object goal category is present in the local map, go to it
            category_map = map_features[e, goal_category[e] + 8, :, :]
            if (category_map == 1).sum() > 0:
                goal_map[e] = category_map == 1

            # Else, set unexplored area as the goal
            else:
                goal_map[e] = map_features[e, 1, :, :] == 0

                goal_map[e] = 1 - skimage.morphology.binary_dilation(
                    1 - goal_map[e],
                    skimage.morphology.disk(10)
                ).astype(int)

                # Select the frontier
                goal_map[e] = (
                    skimage.morphology.binary_dilation(
                        goal_map[e],
                        skimage.morphology.disk(1)
                    ).astype(int) - goal_map[e]
                )

        goal_map = torch.from_numpy(goal_map)
        regression_logits = None
        return goal_map, regression_logits
