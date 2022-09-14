import os
from pathlib import Path
import argparse
import traceback
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation import MultiAgentEpisode as Episode
from ray.tune.logger import pretty_print

# Ray 2.0.0 imports
# from ray.rllib.algorithms import ppo, ddppo
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.evaluation import Episode
# from ray.tune import tuner
# from ray.tune.tune_config import TuneConfig
# from ray.air.config import RunConfig

from submission.utils.config_utils import get_config
from submission.policy.semantic_exploration_policy_network import SemanticExplorationPolicyModelWrapper
from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


class LoggingCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies,
                        episode: Episode, env_index, **kwargs):
        info = episode.last_info_for()

        for k in ["sparse_goal_reward", "discounted_sparse_goal_reward",
                  "intrinsic_reward", "unscaled_intrinsic_reward", "discounted_unscaled_intrinsic_reward",
                  "dense_goal_reward", "unscaled_dense_goal_reward", "discounted_unscaled_dense_goal_reward"]:
            if k not in info:
                continue
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
        default="submission/configs/debug_config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    print("-" * 100)
    print("Config:")
    config, config_str = get_config(args.config_path)
    print(config_str)
    print("-" * 100)
    print()

    print("Cluster resources:")
    ip_head = os.environ.get("ip_head")
    redis_password = os.environ.get("redis_password")
    print(f"ip_head: {ip_head}")
    print(f"redis_password: {redis_password}")
    if ip_head is not None and redis_password is not None:
        try:
            ray.init(address=ip_head, _redis_password=redis_password)
        except:
            print("Could not initialize cluster with "
                  "ray.init(address=ip_head, _redis_password=redis_password). "
                  "Initializing it with ray.init()")
            print()
            traceback.print_exc()
            ray.init()
    else:
        ray.init()
    print(f"ray.nodes(): {ray.nodes()}")
    print(f"ray.cluster_resources(): {ray.cluster_resources()}")
    print("-" * 100)
    print()

    ModelCatalog.register_custom_model(
        "semexp_custom_model",
        SemanticExplorationPolicyModelWrapper
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
            "custom_model": "semexp_custom_model",
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
        "rollout_fragment_length": config.TRAIN.RL.rollout_fragment_length,
        "num_sgd_iter": config.TRAIN.RL.sgd_epochs,
        "framework": "torch",
        # "disable_env_checking": True,     # Ray 2.0.0
        "normalize_actions": False,
        "_disable_preprocessor_api": True,
        # "ignore_worker_failures": True,
        # "recreate_failed_workers": True,  # Ray 2.0.0
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
            #    in sgd_minibatch_size sized pieces)
            "train_batch_size": (config.TRAIN.RL.rollout_fragment_length *
                                 config.TRAIN.RL.PPO.num_workers),
            "sgd_minibatch_size": 2 * config.TRAIN.RL.rollout_fragment_length,
        })
    elif config.TRAIN.RL.algorithm == "DDPPO":
        train_config.update({
            # Workers
            "num_workers": config.TRAIN.RL.DDPPO.num_workers,
            "num_envs_per_worker": config.TRAIN.RL.DDPPO.num_envs_per_worker,
            "num_gpus_per_worker": config.TRAIN.RL.DDPPO.num_gpus_per_worker,
            "num_cpus_per_worker": config.TRAIN.RL.DDPPO.num_cpus_per_worker,
            "remote_worker_envs": config.TRAIN.RL.DDPPO.remote_worker_envs,
            "remote_env_batch_wait_ms": config.TRAIN.RL.DDPPO.remote_env_batch_wait_ms,
            # Batching
            #   train_batch_size: total batch size is implicitly
            #    (num_workers * num_envs_per_worker * rollout_fragment_length)
            #   sgd_minibatch_size: total SGD minibatch size is
            #    (num_workers * sgd_minibatch_size)
            "sgd_minibatch_size": max(
                2 * config.TRAIN.RL.rollout_fragment_length //
                config.TRAIN.RL.DDPPO.num_workers,
                1
            ),
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

    # Training with class API (Ray 2.0.0)
    # tuner = tuner.Tuner(
    #     config.TRAIN.RL.algorithm,
    #     param_space=train_config,
    #     run_config=RunConfig(name=config.TRAIN.RL.exp_name),
    #     tune_config=TuneConfig(max_concurrent_trials=1)
    # )
    # tuner.fit()

    # Training with functional API (Ray 1.8.0)
    ray.tune.run(
        config.TRAIN.RL.algorithm,
        name=config.TRAIN.RL.exp_name,
        config=train_config,
        max_concurrent_trials=1,
        checkpoint_freq=config.TRAIN.RL.checkpoint_freq,
        restore=config.TRAIN.RL.restore,
    )

    ray.shutdown()
