import gzip
import json
import os
from typing import List, Dict
import glob

from habitat.config import Config
from habitat.core.dataset import Dataset, ALL_SCENES_MASK
from habitat.core.registry import registry
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ObjectGoal

from submission.utils.constants import (
    challenge_goal_name_to_goal_name,
    goal_id_to_goal_name
)


@registry.register_dataset(name="SemexpPolicyTraining")
class SemanticExplorationPolicyTrainingDataset(Dataset):
    """
    Simple dataset used to train the semantic exploration policy spawning
    the agent at a random location in the scene with a random object goal.
    """
    episodes: List[ObjectGoalNavEpisode]
    category_to_task_category_id: Dict[str, int]

    def __init__(self, config: Config, dataset_generation: bool = False):
        self.episodes = []

        goal_name_to_challenge_goal_name = {
            v: k for k, v in challenge_goal_name_to_goal_name.items()
        }
        self.category_to_task_category_id = {
            goal_name_to_challenge_goal_name[goal_name]: goal_id
            for goal_id, goal_name in goal_id_to_goal_name.items()
        }

        if dataset_generation:
            return

        # Read file for all split
        split_filepath = config.DATA_PATH.format(
            split=config.SPLIT,
            dataset_type=config.DATASET_TYPE,
            scene_type=config.SCENE_TYPE
        )
        with gzip.open(split_filepath, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        split_dir = os.path.dirname(split_filepath)
        if os.path.exists(f"{split_dir}/scenes"):
            for scene_path in glob.glob(f"{split_dir}/scenes/*"):
                scene_id = scene_path.split("/")[-1].split(".")[0]
                if (ALL_SCENES_MASK not in config.CONTENT_SCENES
                        and scene_id not in config.CONTENT_SCENES):
                    continue
                with gzip.open(scene_path, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

    def from_json(self, json_str: str, scenes_dir: str):
        for episode in json.loads(json_str)["episodes"]:
            episode = ObjectGoalNavEpisode(**episode)
            episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.goals = [ObjectGoal(**goal) for goal in episode.goals]
            self.episodes.append(episode)
