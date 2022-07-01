import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import Policy
from .utils.distributions import DiagGaussian
from .utils.model import Flatten, NNBase


class SemanticExplorationPolicy(Policy):
    """
    This predicts a high-level goal from map features.
    """

    def __init__(self, config):
        super().__init__(config)

        self._goal_update_steps = config.AGENT.POLICY.SEMANTIC.goal_update_steps
        self.training_downscaling = config.AGENT.POLICY.SEMANTIC.training_downscaling
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        num_sem_categories = config.ENVIRONMENT.num_sem_categories
        local_map_size = (
            config.AGENT.SEMANTIC_MAP.map_size_cm //
            config.AGENT.SEMANTIC_MAP.global_downscaling //
            self.map_resolution //
            self.training_downscaling
        )
        map_features_shape = (
            config.ENVIRONMENT.num_sem_categories + 8,
            local_map_size,
            local_map_size
        )
        num_outputs = 2
        hidden_size = 256
        self.deterministic = False

        self.network = Goal_Oriented_Semantic_Policy(
            map_features_shape, hidden_size, num_sem_categories
        )
        self.dist = DiagGaussian(self.network.output_size, num_outputs)

        state_dict = torch.load("submission/policy/semantic_exploration_policy.pth",
                                map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    @property
    def goal_update_steps(self):
        return self._goal_update_steps

    def explore_otherwise(self,
                          map_features,
                          local_pose,
                          goal_category,
                          goal_map,
                          found_goal,
                          found_hint):
        batch_size, goal_map_size, _ = goal_map.shape

        orientation = torch.div(torch.trunc(local_pose[:, 2]) % 360, 5).long()
        map_features = F.avg_pool2d(map_features, self.training_downscaling)

        dist = self.dist(self.network(map_features, orientation, goal_category))

        if self.deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        goal_location = (nn.Sigmoid()(action) * (goal_map_size - 1)).long()

        for e in range(batch_size):
            if not found_goal[e] and not found_hint[e]:
                goal_map[e, goal_location[e, 0], goal_location[e, 1]] = 1

        return goal_map


class Goal_Oriented_Semantic_Policy(NNBase):
    def __init__(self, map_features_shape, hidden_size, num_sem_categories):
        super(Goal_Oriented_Semantic_Policy, self).__init__(False, hidden_size, hidden_size)

        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)

        out_size = int(map_features_shape[1] / 16.0) * int(map_features_shape[2] / 16.0)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(map_features_shape[0], 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
        )

        self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)

    def forward(self, map_features, orientation, object_goal):
        map_features = self.main(map_features)
        orientation_emb = self.orientation_emb(orientation)
        goal_emb = self.goal_emb(object_goal)
        x = torch.cat((map_features, orientation_emb, goal_emb), 1)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        return x
