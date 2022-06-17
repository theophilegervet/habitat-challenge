from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch.nn as nn
from torch import Tensor


class Policy(nn.Module, ABC):
    """
    Policy to select high-level goals.
    """

    @abstractmethod
    def forward(self,
                map_features: Tensor,
                goal_category: Tensor
                ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, channels, M, M)
            goal_category: semantic goal category

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size,)
            regression_logits: if we're using a regression policy, tensor
             of pre-sigmoid (y, x) locations to use in MSE loss of shape
              (batch_size, 2)
        """
        pass
