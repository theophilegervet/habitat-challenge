import os
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

import ray
from ray.tune import tuner
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.logger import pretty_print

from submission.utils.config_utils import get_config
from submission.policy.semantic_exploration_policy_rllib import SemanticExplorationPolicyNetwork
from submission.vector_env.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


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


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/config.yaml")

    ray.init(log_to_driver=False)

    ModelCatalog.register_custom_model(
        "semantic_exploration_policy",
        SemanticExplorationPolicyWrapper
    )

    map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
    num_sem_categories = config.ENVIRONMENT.num_sem_categories
    local_map_size = (
        config.AGENT.SEMANTIC_MAP.map_size_cm //
        config.AGENT.SEMANTIC_MAP.global_downscaling //
        map_resolution
    )
    map_features_shape = (
        config.ENVIRONMENT.num_sem_categories + 8,
        local_map_size,
        local_map_size
    )

    train_config = {
        "env": SemanticExplorationPolicyTrainingEnvWrapper,
        "env_config": {"config": config},
        "num_gpus": config.TRAIN.RL.num_gpus,
        "num_gpus_per_worker": config.TRAIN.RL.num_gpus_per_worker,
        "model": {
            "custom_model": "semantic_exploration_policy",
            "custom_model_config": {
                "map_features_shape": map_features_shape,
                "hidden_size": 256,
                "num_sem_categories": config.ENVIRONMENT.num_sem_categories,
            }
        },
        "num_workers": config.TRAIN.RL.num_workers,
        "framework": "torch",
        "disable_env_checking": True,
        "_disable_preprocessor_api": True,
        "lr": config.TRAIN.RL.lr,
        "gamma": config.TRAIN.RL.gamma,
        "rollout_fragment_length": config.TRAIN.RL.batch_size,
        "train_batch_size": config.TRAIN.RL.batch_size,
        "sgd_minibatch_size": config.TRAIN.RL.batch_size,
    }

    # Debugging
    # ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(train_config)
    # trainer = ppo.PPOTrainer(
    #     config=ppo_config,
    #     env=SemanticExplorationPolicyTrainingEnvWrapper
    # )
    # while True:
    #     result = trainer.train()
    #     print(pretty_print(result))

    tuner = tuner.Tuner(
        "PPO",
        param_space=train_config,
    )
    results = tuner.fit()

    ray.shutdown()
