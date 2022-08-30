from ray.rllib.agents import ppo

from submission.utils.config_utils import get_config
from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper

config, config_str = get_config("submission/configs/debug_config.yaml")
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update({
    "env_config": {"config": config},
    "num_gpus_per_worker": 1
})
algo = ppo.PPOTrainer(
    config=ppo_config,
    env=SemanticExplorationPolicyTrainingEnvWrapper
)
algo.restore("~/ray_results/ddppo_overfit_challenge/DDPPO_SemanticExplorationPolicyTrainingEnvWrapper_7d280_00000_0_2022-08-28_14-29-25/checkpoint_000400/checkpoint-400")
