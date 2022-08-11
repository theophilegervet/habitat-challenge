from submission.utils.config_utils import get_config
from submission.vector_env.semexp_policy_training_env_wrapper import SemexpPolicyTrainingEnvWrapper

config, config_str = get_config("submission/configs/config.yaml")
env = SemexpPolicyTrainingEnvWrapper(config)
