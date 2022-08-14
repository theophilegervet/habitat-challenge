import numpy as np

from submission.utils.config_utils import get_config
from submission.vector_env.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


config, config_str = get_config("submission/configs/config.yaml")
env = SemanticExplorationPolicyTrainingEnvWrapper(config)

obs = env.reset()
print(obs["map_features"].sum())
action = np.array([0.9, 0.9])
obs, reward, done, info = env.step(action)
print(obs.keys())
print(obs["map_features"].sum())
print(reward)
print(done)
print(info)
