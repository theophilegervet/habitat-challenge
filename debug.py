import numpy as np
import warnings
warnings.filterwarnings("ignore")

from submission.utils.config_utils import get_config
from submission.vector_env.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper


config, config_str = get_config("submission/configs/config.yaml")
env = SemanticExplorationPolicyTrainingEnvWrapper(config=config)

obs = env.reset()
print(obs["map_features"].shape)
for _ in range(3):
    action = np.array([0.9, 0.9])
    obs, reward, done, info = env.step(action)
    print(obs["map_features"].shape)
