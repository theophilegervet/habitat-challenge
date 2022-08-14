from submission.utils.config_utils import get_config
from habitat.core.env import Env, RLEnv
from submission.vector_env.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper

config, config_str = get_config("submission/configs/config.yaml")
# env = RLEnv(config=config.TASK_CONFIG)
env = SemanticExplorationPolicyTrainingEnvWrapper(config)

obs = env.reset()
print(obs.keys())
print(obs["local_pose"])
print(obs["goal_category"])
print(obs["map_features"].shape)
