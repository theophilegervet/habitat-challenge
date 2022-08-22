import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray
from ray.rllib.algorithms import ppo, ddppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.evaluation import Episode
from ray.tune import tuner
from ray.tune.logger import pretty_print
from ray.tune.tune_config import TuneConfig
from ray.air.config import RunConfig

from submission.utils.config_utils import get_config
from submission.policy.semantic_exploration_policy_rllib import SemanticExplorationPolicyNetwork
from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


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


class LoggingCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies,
                        episode: Episode, env_index, **kwargs):
        info = episode.last_info_for()

        for k in ["goal_reward", "intrinsic_reward", "unscaled_intrinsic_reward",
                  "discounted_goal_reward", "discounted_unscaled_intrinsic_reward"]:
            if k not in episode.custom_metrics:
                episode.custom_metrics[k] = info[k]
            else:
                episode.custom_metrics[k] += info[k]

        for k in ["action_0", "action_1"]:
            if k not in episode.hist_data:
                episode.hist_data[k] = [info[k]]
            else:
                episode.hist_data[k].append(info[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="submission/configs/ppo_train_challenge_dataset_config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    print("-" * 100)
    print("Config:")
    config, config_str = get_config(args.config_path)
    print(config_str)
    print("-" * 100)

    print("Cluster resources:")
    ray.init(address="auto")
    print(ray.nodes())
    print(ray.cluster_resources())
    print("-" * 100)

    ModelCatalog.register_custom_model(
        "semantic_exploration_policy",
        SemanticExplorationPolicyWrapper
    )

    local_map_size = (
        config.AGENT.SEMANTIC_MAP.map_size_cm //
        config.AGENT.SEMANTIC_MAP.global_downscaling //
        config.AGENT.SEMANTIC_MAP.map_resolution //
        config.AGENT.POLICY.SEMANTIC.inference_downscaling
    )
    map_features_shape = (
        config.ENVIRONMENT.num_sem_categories + 8,
        local_map_size,
        local_map_size
    )

    train_config = {
        "env": SemanticExplorationPolicyTrainingEnvWrapper,
        "env_config": {"config": config},
        "callbacks": LoggingCallback,
        "model": {
            "custom_model": "semantic_exploration_policy",
            "custom_model_config": {
                "map_features_shape": map_features_shape,
                "hidden_size": 256,
                "num_sem_categories": config.ENVIRONMENT.num_sem_categories,
            }
        },
        "gamma": config.TRAIN.RL.gamma,
        "lr": config.TRAIN.RL.lr,
        "entropy_coeff": config.TRAIN.RL.entropy_coeff,
        "clip_param": config.TRAIN.RL.clip_param,
        "framework": "torch",
        "disable_env_checking": True,
        "_disable_preprocessor_api": True,
    }

    if config.TRAIN.RL.algorithm == "PPO":
        train_config.update({
            # Workers
            "num_workers": config.TRAIN.RL.PPO.num_workers,
            "num_gpus": config.TRAIN.RL.PPO.num_gpus,
            "num_cpus_for_driver": config.TRAIN.RL.PPO.num_cpus_for_driver,
            "num_gpus_per_worker": config.TRAIN.RL.PPO.num_gpus_per_worker,
            "num_cpus_per_worker": config.TRAIN.RL.PPO.num_cpus_per_worker,
            # Batching
            #   train_batch_size: total batch size
            #   sgd_minibatch_size: SGD minibatch size (chunk train_batch_size
            #    in sgd_minibatch_size pieces)
            "rollout_fragment_length": config.TRAIN.RL.PPO.rollout_fragment_length,
            "train_batch_size": (config.TRAIN.RL.PPO.rollout_fragment_length *
                                 config.TRAIN.RL.PPO.num_workers),
            "sgd_minibatch_size": 2 * config.TRAIN.RL.PPO.rollout_fragment_length,
            "num_sgd_iter": config.TRAIN.RL.PPO.sgd_epochs,
        })
    elif config.TRAIN.RL.algorithm == "DDPPO":
        train_config.update({
            "num_workers": config.TRAIN.RL.DDPPO.num_workers,
            "num_envs_per_worker": config.TRAIN.RL.DDPPO.num_envs_per_worker,
            "remote_worker_envs": True,
            "num_gpus_per_worker": config.TRAIN.RL.DDPPO.num_gpus_per_worker,
            "remote_env_batch_wait_ms": config.TRAIN.RL.DDPPO.remote_env_batch_wait_ms,
            "num_sgd_iter": config.TRAIN.RL.DDPPO.sgd_steps_per_batch,
            "sgd_minibatch_size": config.TRAIN.RL.DDPPO.minibatch_size,
            "rollout_fragment_length": (config.TRAIN.RL.DDPPO.sgd_steps_per_batch *
                                        config.TRAIN.RL.DDPPO.minibatch_size //
                                        config.TRAIN.RL.DDPPO.num_envs_per_worker)
        })

    # Debugging
    # if config.TRAIN.RL.algorithm == "PPO":
    #     ppo_config = ppo.DEFAULT_CONFIG.copy()
    #     ppo_config.update(train_config)
    #     trainer = ppo.PPO(
    #         config=ppo_config,
    #         env=SemanticExplorationPolicyTrainingEnvWrapper
    #     )
    # elif config.TRAIN.RL.algorithm == "DDPPO":
    #     ddppo_config = ddppo.DEFAULT_CONFIG.copy()
    #     ddppo_config.update(train_config)
    #     trainer = ddppo.DDPPO(
    #         config=ddppo_config,
    #         env=SemanticExplorationPolicyTrainingEnvWrapper
    #     )
    # while True:
    #     result = trainer.train()
    #     print(pretty_print(result))

    # Training
    tuner = tuner.Tuner(
        config.TRAIN.RL.algorithm,
        param_space=train_config,
        run_config=RunConfig(name=config.TRAIN.RL.exp_name),
        tune_config=TuneConfig(max_concurrent_trials=1)
    )
    tuner.fit()

    ray.shutdown()
