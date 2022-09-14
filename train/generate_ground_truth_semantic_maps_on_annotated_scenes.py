import multiprocessing
import tqdm
import torch
import json
import numpy as np
import shutil
import os
import quaternion
import random
import glob
from functools import partial
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import habitat

from submission.utils.config_utils import get_config
from submission.utils.constants import (
    coco_categories,
    hm3d_to_mp3d,
    mp3d_categories_mapping,
    coco_categories_color_palette
)
import submission.utils.pose_utils as pu
from submission.obs_preprocessor.obs_preprocessor import ObsPreprocessor
from submission.semantic_map.semantic_map_state import SemanticMapState
from submission.semantic_map.semantic_map_module import SemanticMapModule


SCENES_ROOT_PATH = (
    Path(__file__).resolve().parent.parent /
    "habitat-challenge-data/data/scene_datasets"
)


class HabitatFloorMaps:
    def __init__(self,
                 sim,
                 generation_method,
                 config,
                 device,
                 num_sampled_points=100000,
                 resolution=5,
                 floor_thr=50,
                 padding=125):
        """
        Arguments:
            generation_method: how to generate semantic maps
            num_sampled_points: number of navigable points to sample to
             create floor maps
            resolution: map bin size (in cm)
            floor_thr: floor threshold (in cm)
            padding: padding to add on map boundaries (in cm)
        """
        assert generation_method in [
            "annotations_top_down",      # use annotation top-down bboxes
            "annotations_first_person",  # use first-person annotations
            "predicted_first_person"     # predict first-person segmentation
        ]

        self.sim = sim
        self.num_sampled_points = num_sampled_points
        self.resolution = resolution
        self.floor_thr = floor_thr
        self.padding = padding

        if generation_method in ["annotations_first_person",
                                 "predicted_first_person"]:
            self.device = device

            self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)
            self.semantic_map = SemanticMapState(config, self.device)
            self.semantic_map_module = SemanticMapModule(config).to(self.device)

        # Sample navigable points
        pts = self._sample_points()

        # Bin points based on x and z values, so that
        # we can quickly pool them based on y-filtering
        self.y_cm = pts[:, 1]
        (
            self.xz_origin_cm,
            self.xz_max_cm,
            self.xz_origin_map,
            self.xz_max_map,
            self.map_size,
            self.xz_centered_cm,
            self.xz_centered_map
        ) = self._make_map(pts[:, [0, 2]])

        # Determine floor heights
        self.floor_heights = self._get_floor_heights()

        # Compute each floor's semantic map from object bounding
        # box annotations
        if generation_method == "annotations_top_down":
            self.floor_semantic_maps = [
                self._get_floor_semantic_map_from_top_down_annotations(floor_height)
                for floor_height in self.floor_heights
            ]
        elif generation_method in ["annotations_first_person",
                                   "predicted_first_person"]:
            self.floor_semantic_maps = [
                self._get_floor_semantic_map_from_first_person(floor_height)
                for floor_height in self.floor_heights
            ]

    def _sample_points(self):
        pts = np.zeros((self.num_sampled_points, 3), dtype=np.float32)
        for i in range(self.num_sampled_points):
            pts[i, :] = self.sim.sample_navigable_point()
        pts = pts * 100.  # m to cm
        return pts

    def _make_map(self, xz):
        xz_origin_cm = np.floor(np.min(xz, axis=0) - self.padding)
        xz_max_cm = np.ceil(np.max(xz, axis=0) + self.padding)

        xz_origin_map = (xz_origin_cm / self.resolution).astype(int)
        map_size = np.ceil((xz_max_cm - xz_origin_cm + 1) / self.resolution).astype(int)
        xz_max_map = xz_origin_map + map_size - 1

        xz_centered_cm = xz - xz_origin_cm
        xz_centered_map = (xz_centered_cm / self.resolution).astype(int)

        return (
            xz_origin_cm,
            xz_max_cm,
            xz_origin_map,
            xz_max_map,
            map_size,
            xz_centered_cm,
            xz_centered_map
        )

    def _get_floor_heights(self):
        floor_heights = []
        hist = np.histogram(
            np.asarray(self.y_cm),
            bins=np.arange(
                self.y_cm.min(), self.y_cm.max() + self.floor_thr, self.floor_thr
            )
        )
        for i in range(len(hist[0])):
            if hist[0][i] > self.num_sampled_points // 5:
                floor_heights.append((hist[1][i] + hist[1][i + 1]) / 2)
        return floor_heights

    def _get_floor_navigable_map(self, y):
        ids = np.logical_and(self.y_cm > y - self.floor_thr,
                             self.y_cm < y + self.floor_thr)
        map = np.zeros((self.map_size[0], self.map_size[1]), dtype=int)
        np.add.at(
            map,
            (
                self.xz_centered_map[ids, 0],
                self.xz_centered_map[ids, 1]
            ),
            1
        )
        map[map > 0] = 1.
        return map

    def _get_floor_semantic_map_from_top_down_annotations(self, y):
        navigable_map = self._get_floor_navigable_map(y)
        sem_map = np.zeros((
            len(coco_categories.keys()),
            navigable_map.shape[0],
            navigable_map.shape[1]
        ))

        # First dimension of semantic map is navigable region
        sem_map[0] = navigable_map

        for obj in self.sim.semantic_annotations().objects:
            category_id = mp3d_categories_mapping.get(
                hm3d_to_mp3d.get(
                    obj.category.name().lower().strip()
                )
            )
            if category_id is None:
                continue

            bbox = [
                obj.aabb.center[0] - obj.aabb.sizes[0] / 2,
                obj.aabb.center[2] - obj.aabb.sizes[2] / 2,
                obj.aabb.center[0] + obj.aabb.sizes[0] / 2,
                obj.aabb.center[2] + obj.aabb.sizes[2] / 2
            ]
            bbox = [int(x * 100. / self.resolution) for x in bbox]
            bbox = [
                bbox[0] - self.xz_origin_map[0],
                bbox[1] - self.xz_origin_map[1],
                bbox[2] - self.xz_origin_map[0],
                bbox[3] - self.xz_origin_map[1]
            ]

            if -0.25 < obj.aabb.center[1] - y / 100. < 1.5:
                sem_map[category_id + 1, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1.
                sem_map[0, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0.

        return sem_map

    def _get_floor_semantic_map_from_first_person(
            self, y, num_frames=10, batch_size=1):
        self.obs_preprocessor.reset()
        self.semantic_map.init_map_and_pose()

        self.obs_preprocessor.set_instance_id_to_category_id(
            torch.tensor([
                mp3d_categories_mapping.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip()),
                    self.obs_preprocessor.num_sem_categories - 1
                )
                for obj in self.sim.semantic_annotations().objects
            ])
        )

        # Regenerate original points on the floor
        xz_cm = self.xz_centered_cm + self.xz_origin_cm
        positions = np.stack([xz_cm[:, 0], self.y_cm, xz_cm[:, 1]], axis=1)
        positions = positions / 100.  # cm to m
        ids = np.logical_and(self.y_cm > y - self.floor_thr,
                             self.y_cm < y + self.floor_thr)
        positions = positions[ids]

        # Subsample num_frames points
        idxs = random.sample(range(len(positions)), num_frames)
        all_positions = positions[idxs]

        # Batch points
        for i in range(0, num_frames, batch_size):
            positions = all_positions[i:i + batch_size]
            sequence_length = positions.shape[0]

            # Sample rotations
            yaws = np.random.random(sequence_length) * 2 * np.pi
            rotations = quaternion.from_euler_angles(0., yaws, 0.)

            seq_obs = [self.sim.get_observations_at(positions[t], rotations[t])
                       for t in range(sequence_length)]
            for t in range(sequence_length):
                pose = pu.get_pose(positions[t], rotations[t])
                seq_obs[t]["gps"] = np.array([pose[0], -pose[1]])
                seq_obs[t]["compass"] = [pose[2]]

            # Preprocess observations
            (
                seq_obs_preprocessed, seq_semantic_frame, seq_pose_delta, _, _
            ) = self.obs_preprocessor.preprocess_sequence(seq_obs)

            seq_dones = torch.tensor([False] * sequence_length)
            seq_update_global = torch.tensor([True] * sequence_length)

            # Update semantic map with observations
            (
                _,
                self.semantic_map.local_map,
                self.semantic_map.global_map,
                seq_local_pose,
                seq_global_pose,
                seq_lmb,
                seq_origins,
            ) = self.semantic_map_module(
                seq_obs_preprocessed.unsqueeze(0),
                seq_pose_delta.unsqueeze(0).to(self.device),
                seq_dones.unsqueeze(0).to(self.device),
                seq_update_global.unsqueeze(0).to(self.device),
                self.semantic_map.local_map,
                self.semantic_map.global_map,
                self.semantic_map.local_pose,
                self.semantic_map.global_pose,
                self.semantic_map.lmb,
                self.semantic_map.origins,
            )

            self.semantic_map.local_pose = seq_local_pose[:, -1]
            self.semantic_map.global_pose = seq_global_pose[:, -1]
            self.semantic_map.lmb = seq_lmb[:, -1]
            self.semantic_map.origins = seq_origins[:, -1]

        navigable_map = self._get_floor_navigable_map(y)
        sem_map = np.zeros((
            len(coco_categories.keys()),
            navigable_map.shape[0],
            navigable_map.shape[1]
        ))
        sem_map[0] = navigable_map
        x2 = self.semantic_map.global_h // 2 - self.xz_origin_map[0]
        z2 = self.semantic_map.global_w // 2 - self.xz_origin_map[1]
        x1 = x2 - self.map_size[0]
        z1 = z2 - self.map_size[1]
        sem_map[1:] = np.flip(
            self.semantic_map.global_map.cpu().numpy()[0, 4:-1, x1:x2, z1:z2],
            (1, 2)
        )

        return sem_map


def visualize_sem_map(sem_map):
    def compress_sem_map(sem_map):
        c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
        for i in range(sem_map.shape[0]):
            c_map[sem_map[i] > 0.] = i + 1
        return c_map
    c_map = compress_sem_map(sem_map)
    color_palette = [
        1.0, 1.0, 1.0,     # empty space
        0.95, 0.95, 0.95,  # explored area
        *coco_categories_color_palette
    ]
    color_palette = [int(x * 255.) for x in color_palette]
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    return semantic_img


def generate_scene_semantic_maps(scene_path: str,
                                 generation_method: str,
                                 device: torch.device):
    scene_dir = "/".join(scene_path.split("/")[:-1])
    scene_file = scene_path.split("/")[-1]
    scene_id = scene_file.split(".")[0]
    map_dir = scene_dir + f"/floor_semantic_maps_{generation_method}"
    shutil.rmtree(map_dir, ignore_errors=True)
    os.makedirs(map_dir, exist_ok=True)

    config, _ = get_config("submission/configs/generate_dataset_config.yaml")
    config.defrost()
    if generation_method == "annotations_first_person":
        config.GROUND_TRUTH_SEMANTICS = 1
    elif generation_method == "predicted_first_person":
        config.GROUND_TRUTH_SEMANTICS = 0
    task_config = config.TASK_CONFIG
    task_config.SIMULATOR.SCENE = scene_path
    task_config.SIMULATOR.SCENE_DATASET = f"{SCENES_ROOT_PATH}/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    config.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=task_config.SIMULATOR)
    floor_maps = HabitatFloorMaps(sim, generation_method, config, device)

    print(f"Saving {generation_method} floor semantic maps for {scene_dir}")

    for i, sem_map in enumerate(floor_maps.floor_semantic_maps):
        sem_map_vis = visualize_sem_map(sem_map)

        np.save(
            f"{map_dir}/{scene_id}_floor{i}.npy",
            sem_map.astype(bool)
        )
        sem_map_vis.save(
            f"{map_dir}/{scene_id}_floor{i}.png", "PNG"
        )

    with open(f"{map_dir}/{scene_id}_info.json", "w") as f:
        json.dump(
            {
                "floor_heights_cm": [int(x) for x in floor_maps.floor_heights],
                "xz_origin_cm": [int(x) for x in floor_maps.xz_origin_cm],
                "xz_origin_map": [int(x) for x in floor_maps.xz_origin_map],
                "map_size": [int(x) for x in floor_maps.map_size],
                "map_generation_parameters": {
                    "resolution_cm": floor_maps.resolution,
                    "floor_threshold_cm": floor_maps.floor_thr,
                    "padding_cm": floor_maps.padding
                }
            },
            f, indent=4
        )

    sim.close()


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    for split in ["val"]:
        # For scenes with semantic annotations, generate semantic maps
        # from top-down bounding boxes
        scenes = glob.glob(f"{SCENES_ROOT_PATH}/hm3d/{split}/*/*semantic.glb")
        scenes = [scene.replace("semantic.glb", "basis.glb") for scene in scenes]
        generate_annotations_top_down = partial(
            generate_scene_semantic_maps,
            generation_method="annotations_top_down",
            device=torch.device("cuda:1")
        )
        for scene in scenes:
            generate_annotations_top_down(scene)
            break

        # For scenes all scenes, generate semantic maps from first-person
        # segmentation predictions
        scenes = glob.glob(f"{SCENES_ROOT_PATH}/hm3d/{split}/*/*basis.glb")
        generate_predicted_first_person = partial(
            generate_scene_semantic_maps,
            generation_method="predicted_first_person",
            device=torch.device("cuda:1")
        )
        for scene in scenes:
            generate_predicted_first_person(scene)
            break

        # with multiprocessing.Pool(12) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        #     for _ in pool.imap_unordered(generate_scene_ground_truth_maps, scenes):
        #         pbar.update()
