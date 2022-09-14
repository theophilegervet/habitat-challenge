import glob
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import tqdm
import numpy as np
import quaternion
from functools import partial
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

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
    "habitat-challenge-data/custom_objectgoal_hm3d"
)


def generate_episode(sim,
                     episode_count: int,
                     scene_info: dict
                     ) -> ObjectGoalNavEpisode:
    print(scene_info.keys())
    print(len(scene_info["floor_maps"]))
    print(scene_info["floor_maps"][0].shape)
    raise NotImplementedError

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


def generate_scene_episodes(scene_path: str,
                            dataset_type: str,
                            split: str,
                            num_episodes: int):
    assert dataset_type in ["annotated_scenes", "unannotated_scenes"]
    if dataset_type == "annotated_scenes":
        semantic_map_type = "annotations_top_down"
    else:
        semantic_map_type = "predicted_first_person"

    config = habitat.get_config(str(
        Path(__file__).resolve().parent.parent /
        f"submission/dataset/train_custom_{dataset_type}_dataset_config.yaml"
    ))
    config.defrost()
    config.SIMULATOR.SCENE = scene_path
    config.SIMULATOR.SCENE_DATASET = f"{SCENES_ROOT_PATH}/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    config.DATASET.SPLIT = split
    config.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)

    # Load scene floor semantic maps
    scene_dir = "/".join(scene_path.split("/")[:-1])
    scene_key = scene_path.split("/")[-1].split(".")[0]
    map_dir = scene_dir + f"/floor_semantic_maps_{semantic_map_type}"
    print(f"{map_dir}/{scene_key}_info.json")
    with open(f"{map_dir}/{scene_key}_info.json", "r") as f:
        scene_info = json.load(f)
    scene_info["floor_maps"] = []
    for i in range(len(scene_info["floor_heights"])):
        sem_map = np.load(f"{map_dir}/{scene_key}_floor{i}.npy")
        scene_info["floor_maps"].append(sem_map)

    # Create dataset and episodes
    dataset = SemanticExplorationPolicyTrainingDataset(
        config.DATASET, dataset_generation=True)
    for episode_count in range(num_episodes):
        episode = generate_episode(sim, episode_count, scene_info)
        dataset.episodes.append(episode)
        raise NotImplementedError  # TODO
    for ep in dataset.episodes:
        ep.scene_id = ep.scene_id.split("scene_datasets/")[-1]

    sim.close()

    # Store episodes with one file per scene
    out_path = f"{DATASET_ROOT_PATH}/{split}/scenes/{scene_key}.json.gz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write(dataset.to_json())


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    # Dataset is stored in per-scene files, generate empty split files
    for split in ["val", "train"]:
        for dataset_type in ["annotated_scenes", "unannotated_scenes"]:
            split_filepath = f"{DATASET_ROOT_PATH}/{dataset_type}/{split}/{split}.json.gz"
            os.makedirs(os.path.dirname(split_filepath), exist_ok=True)
            with gzip.open(split_filepath, "wt") as f:
                json.dump(dict(episodes=[]), f)

    # Generate per-scene files
    for split in ["val"]:
        # For scenes with semantic annotations, generate episode dataset
        # from semantic maps built from top-down bounding boxes
        scenes = glob.glob(f"{SCENES_ROOT_PATH}/hm3d/{split}/*/*semantic.glb")
        scenes = [scene.replace("semantic.glb", "basis.glb") for scene in scenes]
        generate_annotated_scene_episodes = partial(
            generate_scene_episodes,
            dataset_type="annotated_scenes",
            split=split,
            # 100 annotated train scenes * 40K ep per scene = 4M train ep
            # 20 annotated val scenes * 100 ep per scene = 2K eval ep
            num_episodes=40000 if split == "train" else 100
        )
        # with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        #     for _ in pool.imap_unordered(generate_annotated_scene_episodes, scenes):
        #         pbar.update()
        generate_annotated_scene_episodes(scenes[0])
        raise NotImplementedError  # TODO

        # For all scenes, generate episode dataset from semantic maps
        # built from from first-person segmentation predictions
        scenes = glob.glob(f"{SCENES_ROOT_PATH}/hm3d/{split}/*/*basis.glb")
        generate_unannotated_scene_episodes = partial(
            generate_scene_episodes,
            dataset_type="unannotated_scenes",
            split=split,
            # 800 unannotated train scenes * 5K ep per scene = 4M train ep
            # 100 unannotated val scenes * 20 ep per scene = 2K eval ep
            num_episodes=5000 if split == "train" else 20
        )
        # with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        #     for _ in pool.imap_unordered(generate_unannotated_scene_episodes, scenes):
        #         pbar.update()
