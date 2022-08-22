import glob
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import tqdm
import numpy as np
import quaternion
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import habitat
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ObjectGoal

from submission.dataset.semexp_policy_training_dataset import SemanticExplorationPolicyTrainingDataset
from submission.utils.constants import challenge_goal_name_to_goal_name


SCENES_ROOT_PATH = (
    Path(__file__).resolve().parent.parent /
    "habitat-challenge-data/data/scene_datasets"
)
DATASET_ROOT_PATH = (
    Path(__file__).resolve().parent.parent /
    "habitat-challenge-data/semexp_policy_training_hm3d"
)


def generate_episode(sim, episode_count: int) -> ObjectGoalNavEpisode:
    # Position
    start_position = sim.pathfinder.get_random_navigable_point()
    attempt = 1
    while sim.pathfinder.distance_to_closest_obstacle(start_position) < 1.0 and attempt < 50:
        start_position = sim.pathfinder.get_random_navigable_point()
        attempt += 1

    # Rotation
    start_yaw = random.random() * 2 * np.pi
    start_rotation = quaternion.from_euler_angles(0, start_yaw, 0)
    start_rotation = quaternion.as_float_array(start_rotation)
    start_rotation = [0., start_rotation[0], 0., start_rotation[2]]

    # Object goal
    object_category = random.choice(list(challenge_goal_name_to_goal_name.keys()))

    return ObjectGoalNavEpisode(
        episode_id=str(episode_count),
        scene_id=sim.habitat_config.SCENE,
        start_position=start_position,
        start_rotation=start_rotation,
        object_category=object_category,
        goals=[ObjectGoal(position=[], object_id="")]
    )


def generate_scene_episodes(scene_path: str, num_episodes: int = 5000):
    # 800 train scenes * 5K episodes per scene = 4M train episodes
    # 100 val scenes * 100 episodes per scene = 1K eval episodes
    if "train" in scene_path:
        split = "train"
    elif "val" in scene_path:
        split = "val"
    else:
        raise ValueError

    config = habitat.get_config(str(
        Path(__file__).resolve().parent.parent /
        "submission/dataset/ppo_custom_dataset_config.yaml"
    ))
    config.defrost()
    config.SIMULATOR.SCENE = scene_path
    config.DATASET.SPLIT = split
    config.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)

    # Create dataset and episodes
    dataset = SemanticExplorationPolicyTrainingDataset(
        config.DATASET, dataset_generation=True)
    for episode_count in range(num_episodes):
        dataset.episodes.append(generate_episode(sim, episode_count))
    for ep in dataset.episodes:
        ep.scene_id = ep.scene_id.split("scene_datasets/")[-1]

    # Store episodes with one file per scene
    scene_key = scene_path.split("/")[-1].split(".")[0]
    out_path = f"{DATASET_ROOT_PATH}/{split}/scenes/{scene_key}.json.gz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write(dataset.to_json())


# Dataset is stored in per-scene files, generate empty split files
for split in ["val", "train"]:
    split_filepath = f"{DATASET_ROOT_PATH}/{split}/{split}.json.gz"
    os.makedirs(os.path.dirname(split_filepath), exist_ok=True)
    with gzip.open(split_filepath, "wt") as f:
        json.dump(dict(episodes=[]), f)

# Generate per-scene files
for split in ["train"]:
    scenes = glob.glob(f"{SCENES_ROOT_PATH}/hm3d/{split}/*/*basis.glb")

    with multiprocessing.Pool(80) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(generate_scene_episodes, scenes):
            pbar.update()
