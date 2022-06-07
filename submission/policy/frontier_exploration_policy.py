from typing import Tuple, Optional
import torch
from torch import Tensor

from .policy import Policy


class FrontierExplorationPolicy(Policy):

    def forward(self,
                map_features: Tensor,
                goal_category: Tensor
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO Implement
        goal_actions = torch.ones(1, 2)
        regression_logits = None
        return goal_actions, regression_logits
