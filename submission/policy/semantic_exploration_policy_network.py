import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from .utils.model import Flatten, NNBase


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


class SemanticExplorationPolicyWrapper(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.policy_network = SemanticExplorationPolicyNetwork(
            model_config["custom_model_config"]["map_features_shape"],
            model_config["custom_model_config"]["hidden_size"],
            model_config["custom_model_config"]["num_sem_categories"]
        )
        self.dummy = nn.Parameter(torch.empty(0))

        self.value = None

    def forward(self, input_dict, state, seq_lens):
        for k, v in input_dict["obs"].items():
            if type(v) == np.ndarray:
                input_dict["obs"][k] = torch.from_numpy(v).to(self.dummy.device)

        orientation = torch.div(
            torch.trunc(input_dict["obs"]["local_pose"][:, 2]) % 360, 5
        ).long()

        outputs, value = self.policy_network(
            input_dict["obs"]["map_features"],
            orientation,
            input_dict["obs"]["goal_category"]
        )
        self.value = value

        return outputs, []

    def value_function(self):
        return self.value
