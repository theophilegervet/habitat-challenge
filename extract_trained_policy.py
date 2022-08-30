from ray.rllib.agents import ppo

from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


algo = ppo.PPOTrainer(
    config=ppo.DEFAULT_CONFIG.copy(),
    env=SemanticExplorationPolicyTrainingEnvWrapper
)
algo.restore("~/ray_results/ddppo_overfit_challenge/DDPPO_SemanticExplorationPolicyTrainingEnvWrapper_7d280_00000_0_2022-08-28_14-29-25/checkpoint_000400/checkpoint-400")
