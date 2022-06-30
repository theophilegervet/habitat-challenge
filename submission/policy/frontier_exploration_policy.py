import torch
import torch.nn as nn
import skimage.morphology

from .policy import Policy
from .utils.morphology import binary_dilation


class FrontierExplorationPolicy(Policy):
    """
    This policy picks the closest non-explored point.
    """

    def __init__(self, config):
        super().__init__(config)

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
    def goal_update_steps(self):
        return 1

    def explore_otherwise(self,
                          map_features,
                          global_pose,
                          goal_category,
                          goal_map,
                          found_goal,
                          found_hint):
        # Select unexplored area
        frontier_map = (map_features[:, [1], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel)

        # Select the frontier
        frontier_map = binary_dilation(
            frontier_map, self.select_border_kernel) - frontier_map

        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e] and not found_hint[e]:
                goal_map[e] = frontier_map[e]

        return goal_map
