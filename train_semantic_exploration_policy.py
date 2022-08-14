import os
import torch
import torch.nn as nn

import ray
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
            model_config["map_features_shape"],
            model_config["hidden_size"],
            model_config["num_sem_categories"]
        )

        self.value = None

    def forward(self, input_dict, state, seq_lens):
        orientation = torch.div(
            torch.trunc(input_dict["obs"]["local_pose"][:, 2]) % 360, 5
        ).long()
        outputs, value = self.policy.network(
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

    ray.init(local_mode=True)

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
        # TODO If env_config needs to be a dict, we might need to pass it differently
        #  Maybe serialize it?
        "env_config": {"config": config},
        "num_gpus": 1,  # int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        "num_gpus_per_worker": 1,
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
        "lr": config.TRAIN.RL.lr
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(train_config)
    trainer = ppo.PPOTrainer(
        config=ppo_config,
        env=SemanticExplorationPolicyTrainingEnvWrapper
    )

    # while True:
    #     result = trainer.train()
    #     print(pretty_print(result))
    #     if result["timesteps_total"] >= config.TRAIN.RL.stop_timesteps:
    #         break

    ray.shutdown()
