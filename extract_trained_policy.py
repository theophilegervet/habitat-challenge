from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from submission.utils.config_utils import get_config
from submission.policy.semantic_exploration_policy_rllib_wrapper import SemanticExplorationPolicyWrapper
from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


config, config_str = get_config("submission/configs/debug_config.yaml")


ModelCatalog.register_custom_model(
    "semantic_exploration_policy",
    SemanticExplorationPolicyWrapper
)


# ppo_config = ppo.DEFAULT_CONFIG.copy()
# ppo_config.update({
#     "env_config": {"config": config},
#     "num_gpus_per_worker": 1
# })


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
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update({
    "env": SemanticExplorationPolicyTrainingEnvWrapper,
    "env_config": {"config": config},
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
    # "disable_env_checking": True,
    "_disable_preprocessor_api": True,
    "ignore_worker_failures": True
    # "recreate_failed_workers": True,
})
ppo_config.update({
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
    "rollout_fragment_length": config.TRAIN.RL.rollout_fragment_length,
    "train_batch_size": (config.TRAIN.RL.rollout_fragment_length *
                         config.TRAIN.RL.PPO.num_workers),
    "sgd_minibatch_size": 2 * config.TRAIN.RL.rollout_fragment_length,
    "num_sgd_iter": config.TRAIN.RL.sgd_epochs,
})


algo = ppo.PPOTrainer(
    config=ppo_config,
    env=SemanticExplorationPolicyTrainingEnvWrapper
)
algo.restore("~/ray_results/ddppo_overfit_challenge/DDPPO_SemanticExplorationPolicyTrainingEnvWrapper_7d280_00000_0_2022-08-28_14-29-25/checkpoint_000400/checkpoint-400")


# import pickle
#
# checkpoint_path = "/private/home/theop123/ray_results/ddppo_overfit_challenge/DDPPO_SemanticExplorationPolicyTrainingEnvWrapper_7d280_00000_0_2022-08-28_14-29-25/checkpoint_000400/checkpoint-400"
#
# with open(checkpoint_path, 'rb') as f:
#     checkpoint = pickle.load(f)
#
# weights = pickle.loads(checkpoint["worker"])

# value = model["worker"]
# weights=pickle.loads(value)
#
# print(weights)
