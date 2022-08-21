import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import Policy
from .utils.model import Flatten, NNBase


class SemanticExplorationPolicy(Policy):
    """
    This predicts a high-level goal from map features.
    """

    def __init__(self, config):
        super().__init__(config)

        self._goal_update_steps = config.AGENT.POLICY.SEMANTIC.goal_update_steps
        self.inference_downscaling = config.AGENT.POLICY.SEMANTIC.inference_downscaling
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        num_sem_categories = config.ENVIRONMENT.num_sem_categories
        self.local_map_size = (
            config.AGENT.SEMANTIC_MAP.map_size_cm //
            config.AGENT.SEMANTIC_MAP.global_downscaling //
            self.map_resolution //
            self.inference_downscaling
        )
        map_features_shape = (
            config.ENVIRONMENT.num_sem_categories + 8,
            self.local_map_size,
            self.local_map_size
        )
        hidden_size = 256
        self.network = SemanticExplorationPolicyNetwork(
            map_features_shape, hidden_size, num_sem_categories
        )

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
        map_features = F.avg_pool2d(map_features, self.inference_downscaling)

        outputs, value = self.network(
            map_features,
            orientation,
            goal_category
        )

        # TODO Sample action from network outputs with RLLib ActionDistribution
        goal_location = (nn.Sigmoid()(outputs[:, :2]) * (goal_map_size - 1)).long()

        for e in range(batch_size):
            if not found_goal[e] and not found_hint[e]:
                goal_map[e, goal_location[e, 0], goal_location[e, 1]] = 1

        return goal_map


class SemanticExplorationPolicyNetwork(NNBase):
    def __init__(self, map_features_shape, hidden_size, num_sem_categories):
        super(SemanticExplorationPolicyNetwork, self).__init__(
            False, hidden_size, hidden_size
        )

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

        self.actor_linear = nn.Linear(256, 2)
        self.goal_action_std = nn.Parameter(torch.zeros(1))
        self.critic_linear = nn.Linear(256, 1)

    def forward(self, map_features, orientation, object_goal):
        map_features = self.main(map_features)
        orientation_emb = self.orientation_emb(orientation)
        goal_emb = self.goal_emb(object_goal)
        x = torch.cat((map_features, orientation_emb, goal_emb), 1)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        goal_action_mean = self.actor_linear(x)
        goal_action_std = self.goal_action_std.expand_as(goal_action_mean)
        outputs = torch.cat([goal_action_mean, goal_action_std], dim=1)
        value = self.critic_linear(x).squeeze(-1)
        return outputs, value
