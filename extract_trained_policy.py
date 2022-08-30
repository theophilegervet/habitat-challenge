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

ppo_config = ppo.DEFAULT_CONFIG.copy()
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
ppo_config.update({
    "env_config": {"config": config},
    "model": {
        "custom_model": "semantic_exploration_policy",
        "custom_model_config": {
            "map_features_shape": map_features_shape,
            "hidden_size": 256,
            "num_sem_categories": config.ENVIRONMENT.num_sem_categories,
        },
    },
    "framework": "torch",
    "_disable_preprocessor_api": True,
    "num_gpus_per_worker": 1,
})

algo = ppo.PPOTrainer(
    config=ppo_config,
    env=SemanticExplorationPolicyTrainingEnvWrapper
)
algo.restore(config.AGENT.POLICY.SEMANTIC.checkpoint_path)
policy = algo.get_policy()
print(type(policy))
model = policy.model
print(type(model))
dist_class = policy.dist_class
print(dist_class)
