from submission.utils.config_utils import get_config
from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


config, config_str = get_config("submission/configs/config.yaml")
env = SemanticExplorationPolicyTrainingEnvWrapper(config=config)
