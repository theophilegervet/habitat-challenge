from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper
from submission.dataset.semexp_policy_training_dataset import SemanticExplorationPolicyTrainingDataset
from submission.utils.config_utils import get_config


config, config_str = get_config("submission/configs/ddppo_train_custom_annotated_scenes_dataset_config.yaml")
config.defrost()
config.TASK_CONFIG.DATASET.SPLIT = "val"
config.freeze()
# dataset = SemanticExplorationPolicyTrainingDataset(config=config.TASK_CONFIG.DATASET)
env = SemanticExplorationPolicyTrainingEnvWrapper(config=config)
obs = env.reset()
goal_action = [10, 10]
obs, reward, done, info = env.step(goal_action)
print(reward)
print(done)
print(info)
goal_action = [10, 10]
obs, reward, done, info = env.step(goal_action)
print(reward)
print(done)
print(info)
