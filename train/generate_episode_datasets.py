import argparse
from typing import Optional
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
import skimage.morphology
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

import habitat
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ObjectGoal

from submission.dataset.semexp_policy_training_dataset import SemanticExplorationPolicyTrainingDataset
from submission.planner.fmm_planner import FMMPlanner
from submission.utils.constants import (
    challenge_goal_name_to_goal_name,
    goal_id_to_goal_name,
    goal_id_to_coco_id
)


SCENES_ROOT_PATH = str(
    Path(__file__).resolve().parent.parent /
    "habitat-challenge-data/data/scene_datasets"
)
DATASET_ROOT_PATH = str(
    Path(__file__).resolve().parent.parent /
    "habitat-challenge-data/custom_objectgoal"
)


def generate_episode(sim,
                     episode_count: int,
                     scene_info: dict
                     ) -> Optional[ObjectGoalNavEpisode]:
    """Attempt generating an episode and return None if failed."""
    # Sample a floor
    assert len(scene_info["floor_maps"]) > 0
    floor_idx = np.random.randint(len(scene_info["floor_maps"]))
    floor_height = scene_info["floor_heights_cm"][floor_idx] / 100
    sem_map = scene_info["floor_maps"][floor_idx]
    xz_origin_cm = scene_info["xz_origin_cm"]
    map_size = scene_info["map_size"]
    map_resolution = scene_info["map_generation_parameters"]["resolution_cm"]
    floor_thr = scene_info["map_generation_parameters"]["floor_threshold_cm"] / 100

    # Sample a goal category present on the floor
    category_counts = sem_map.sum(2).sum(1)
    categories_present = [i for i in goal_id_to_coco_id.keys()
                          if category_counts[goal_id_to_coco_id[i] + 1] > 0]
    assert len(categories_present) > 0
    goal_idx = np.random.choice(categories_present)
    goal_name_to_challenge_goal_name = {
        v: k for k, v in challenge_goal_name_to_goal_name.items()}
    object_category = goal_name_to_challenge_goal_name[
        goal_id_to_goal_name[goal_idx]]

    # Sample a starting position from which we can reach this goal
    selem = skimage.morphology.disk(2)
    traversible = skimage.morphology.binary_dilation(sem_map[0], selem) != True
    traversible = 1 - traversible
    planner = FMMPlanner(traversible)
    selem = skimage.morphology.disk(int(100 / map_resolution))
    goal_map = skimage.morphology.binary_dilation(
        sem_map[goal_id_to_coco_id[goal_idx] + 1], selem) != True
    goal_map = 1 - goal_map
    planner.set_multi_goal(goal_map)
    m1 = sem_map[0] > 0
    m2 = planner.fmm_dist > 10.0
    m3 = planner.fmm_dist < 2000.0
    possible_start_positions = np.logical_and(m1, m2)
    possible_start_positions = np.logical_and(possible_start_positions, m3) * 1.0
    if possible_start_positions.sum() == 0:
        print(f"No valid starting position for {object_category}")
        return
    start_position_found = False
    attempts, attempt = 100, 0
    while not start_position_found and attempt < attempts:
        attempt += 1
        start_position = sim.sample_navigable_point()
        if abs(start_position[1] - floor_height) > floor_thr:
            continue
        map_x = int((start_position[0] * 100. - xz_origin_cm[0]) / map_resolution)
        map_x = min(map_x, map_size[0])
        map_z = int((start_position[2] * 100. - xz_origin_cm[1]) / map_resolution)
        map_z = min(map_z, map_size[1])
        if possible_start_positions[map_x, map_z] == 1:
            start_position_found = True
        else:
            continue

    # Sample a starting orientation
    start_yaw = random.random() * 2 * np.pi
    start_rotation = quaternion.from_euler_angles(0, start_yaw, 0)
    start_rotation = quaternion.as_float_array(start_rotation)
    start_rotation = [0., start_rotation[0], 0., start_rotation[2]]

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
                            scene_type: str,
                            split: str,
                            num_episodes: int):
    assert scene_type in ["annotated_scenes", "unannotated_scenes"]
    if scene_type == "annotated_scenes":
        semantic_map_type = "annotations_top_down"
    else:
        semantic_map_type = "predicted_first_person"

    config = habitat.get_config(str(
        Path(__file__).resolve().parent.parent /
        f"submission/dataset/train_custom_{dataset_type}_{scene_type}_dataset_config.yaml"
    ))
    config.defrost()
    config.SIMULATOR.SCENE = scene_path
    if dataset_type == "hm3d":
        config.SIMULATOR.SCENE_DATASET = f"{SCENES_ROOT_PATH}/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    config.DATASET.SPLIT = split
    config.freeze()

    try:
        sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)
    except:
        print(f"Could not create sim for scene {scene_path}")
        return

    # Load scene floor semantic maps
    try:
        if dataset_type in ["hm3d", "mp3d"]:
            scene_dir = "/".join(scene_path.split("/")[:-1])
            scene_key = scene_path.split("/")[-1].split(".")[0]
        elif dataset_type == "gibson":
            scene_dir = ".".join(scene_path.split(".")[:-1])
            scene_key = scene_dir.split("/")[-1]
        map_dir = scene_dir + f"/floor_semantic_maps_{semantic_map_type}"
        with open(f"{map_dir}/{scene_key}_info.json", "r") as f:
            scene_info = json.load(f)
        assert len(scene_info["floor_heights_cm"]) > 0
        selected_sem_maps, selected_floor_heights = [], []
        for i in range(len(scene_info["floor_heights_cm"])):
            sem_map = np.load(f"{map_dir}/{scene_key}_floor{i}.npy")
            category_counts = sem_map.sum(2).sum(1)
            goal_categories_present = [
                i for i in goal_id_to_coco_id.keys()
                if category_counts[goal_id_to_coco_id[i] + 1] > 0
            ]
            if len(goal_categories_present) == 0:
                continue
            selected_sem_maps.append(sem_map)
            selected_floor_heights.append((scene_info["floor_heights_cm"][i]))
        scene_info["floor_maps"] = selected_sem_maps
        scene_info["floor_heights_cm"] = selected_floor_heights
        assert len(scene_info["floor_maps"]) > 0
    except:
        print(f"Could not load floor semantic maps for scene {scene_key}")
        return

    # Create dataset and episodes
    dataset = SemanticExplorationPolicyTrainingDataset(
        config.DATASET, dataset_generation=True)
    while len(dataset.episodes) < num_episodes:
        episode = generate_episode(sim, len(dataset.episodes), scene_info)
        if episode is not None:
            dataset.episodes.append(episode)
    for ep in dataset.episodes:
        ep.scene_id = ep.scene_id.split("scene_datasets/")[-1]

    sim.close()

    # Store episodes with one file per scene
    out_path = (f"{DATASET_ROOT_PATH}_{dataset_type}/{scene_type}/"
                f"{split}/scenes/{scene_key}.json.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write(dataset.to_json())


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    parser = argparse.ArgumentParser("Generate episodes from floor semantic maps.")
    parser.add_argument("--dataset", type=str, default="hm3d",
                        help="Dataset in ['hm3d', 'mp3d', 'gibson'].")
    parser.add_argument("--split", type=str, default="train",
                        help="Split in ['train', 'val'].")
    parser.add_argument("--scene-type", type=str, default="annotated",
                        help="Type of scene to process in ['annotated', 'unannotated'].")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    assert args.dataset in ["hm3d", "mp3d", "gibson"]
    assert args.split in ["train", "val"]
    assert args.scene_type in ["annotated", "unannotated"]
    if args.dataset in ["mp3d", "gibson"]:
        args.split = "train"

    # Dataset is stored in per-scene files, generate empty split files
    split_filepath = (f"{DATASET_ROOT_PATH}_{args.dataset}/"
                      f"{args.scene_type}_scenes/{args.split}/{args.split}.json.gz")
    os.makedirs(os.path.dirname(split_filepath), exist_ok=True)
    with gzip.open(split_filepath, "wt") as f:
        json.dump(dict(episodes=[]), f)

    # Generate per-scene files
    # For scenes with semantic annotations, generate episode dataset
    # from semantic maps built from top-down bounding boxes
    if args.scene_type == "annotated":
        if args.dataset == "hm3d":
            scenes = glob.glob(f"{SCENES_ROOT_PATH}/{args.dataset}/{args.split}/*/*semantic.glb")
            scenes = [scene.replace("semantic.glb", "basis.glb") for scene in scenes]
        elif args.dataset == "mp3d":
            scenes = glob.glob(f"{SCENES_ROOT_PATH}/{args.dataset}/*/*.glb")
        elif args.dataset == "gibson":
            scenes = glob.glob(f"{SCENES_ROOT_PATH}/{args.dataset}/*.scn")
            scenes = [scene.replace(".scn", ".glb") for scene in scenes]

        generate_annotated_scene_episodes = partial(
            generate_scene_episodes,
            dataset_type=args.dataset,
            scene_type="annotated_scenes",
            split=args.split,
            # 4M train ep, 2K val ep
            num_episodes=int(4e6) // len(scenes) if args.split == "train" else 2000 // len(scenes)
        )
        if args.debug:
            generate_annotated_scene_episodes(scenes[0])
        else:
            with multiprocessing.Pool(40) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
                for _ in pool.imap_unordered(generate_annotated_scene_episodes, scenes):
                    pbar.update()

    # For all scenes, generate episode dataset from semantic maps
    # built from first-person segmentation predictions
    if args.scene_type == "unannotated":
        if args.dataset in ["mp3d", "gibson"]:
            raise NotImplementedError

        scenes = glob.glob(f"{SCENES_ROOT_PATH}/{args.dataset}/{args.split}/*/*basis.glb")
        generate_unannotated_scene_episodes = partial(
            generate_scene_episodes,
            dataset_type=args.dataset,
            scene_type="unannotated_scenes",
            split=args.split,
            # 4M train ep, 2K val ep
            num_episodes=int(4e6) // len(scenes) if args.split == "train" else 2000 // len(scenes)
        )
        if args.debug:
            generate_unannotated_scene_episodes(scenes[0])
        else:
            with multiprocessing.Pool(40) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
                for _ in pool.imap_unordered(generate_unannotated_scene_episodes, scenes):
                    pbar.update()
