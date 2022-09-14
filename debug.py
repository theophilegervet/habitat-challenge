from submission.env_wrapper.semexp_policy_training_env_wrapper import SemanticExplorationPolicyTrainingEnvWrapper
from submission.utils.config_utils import get_config


config, config_str = get_config("submission/configs/ddppo_train_custom_annotated_scenes_dataset_config.yaml")
config.defrost()
config.TASK_CONFIG.DATASET.SPLIT = "val"
config.freeze()
env = SemanticExplorationPolicyTrainingEnvWrapper(config=config)
