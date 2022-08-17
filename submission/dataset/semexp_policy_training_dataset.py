import gzip
import json
import os
from typing import List
import glob

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry


@registry.register_dataset(name="SemexpPolicyTraining")
class SemanticExplorationPolicyTrainingDataset(Dataset):
    """
    Simple dataset used to train the semantic exploration policy spawning
    the agent at a random location in the scene.
    """
    episodes: List[Episode]

    def __init__(self, config: Config, dataset_generation: bool = False):
        self.episodes = []

        if dataset_generation:
            return

        # Read file for all split
        split_filepath = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(split_filepath, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        split_dir = os.path.dirname(split_filepath)
        if os.path.exists(f"{split_dir}/scenes"):
            for scene_path in glob.glob(f"{split_dir}/scenes/*"):
                with gzip.open(scene_path, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

    def from_json(self, json_str: str, scenes_dir: str):
        for episode in json.loads(json_str)["episodes"]:
            episode = Episode(**episode)
            episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            self.episodes.append(episode)
