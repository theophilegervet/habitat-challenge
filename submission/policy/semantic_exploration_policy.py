from typing import Tuple, Optional
from torch import Tensor

from .policy import Policy


class SemanticExplorationPolicy(Policy):

    def forward(self,
                map_features: Tensor,
                goal_category: Tensor
                ) -> Tuple[Tensor, Optional[Tensor]]:
        pass
