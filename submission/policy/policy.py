from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch.nn as nn
from torch import Tensor


class Policy(nn.Module, ABC):
    """
    Policy to select high-level goals.
    """

    def __init__(self):
        super(Policy, self).__init__()

    @abstractmethod
    def forward(self, map_features: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Arguments:
            map_features: semantic map features of shape (batch_size, channels, M, M)

        Returns:
            goal_actions: (y, x) goals in [0, 1] x [0, 1] of
             shape (batch_size, 2)
            regression_logits: if we're using a regression policy, tensor
             of pre-sigmoid (y, x) locations to use in MSE loss of shape
              (batch_size, 2)
        """
        pass
