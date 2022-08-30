import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from .semantic_exploration_policy_rllib import SemanticExplorationPolicyNetwork


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
