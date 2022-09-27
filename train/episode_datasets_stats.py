from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from submission.utils.config_utils import get_config
from submission.dataset.semexp_policy_training_dataset import SemanticExplorationPolicyTrainingDataset


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/ddppo_train_custom_hm3d_annotated_scenes_dataset_config.yaml")

    for dataset_type in ["hm3d", "gibson", "mp3d"]:
        config.defrost()
        config.TASK_CONFIG.DATASET.DATASET_TYPE = dataset_type
        config.TASK_CONFIG.DATASET.SCENE_TYPE = "annotated"
        config.TASK_CONFIG.DATASET.SPLIT = "train"
        config.freeze()

        dataset = SemanticExplorationPolicyTrainingDataset(config=config.TASK_CONFIG.DATASET)

        goal_counter = Counter()
        euclidean_distances_to_goal = []

        for ep in dataset.episodes:
            goal_counter[ep.object_category] += 1

        print(dataset_type)
        print(len(dataset.episodes))
        print(dict(goal_counter))
        print()
