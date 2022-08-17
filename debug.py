from train.semexp_policy_training_dataset import SemanticExplorationPolicyTrainingDataset
import habitat
from habitat.core.env import Env


config = habitat.get_config("train/semexp_policy_training_env_config.yaml")
env = Env(config)
