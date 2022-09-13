import multiprocessing
import tqdm
import torch
import numpy as np
import quaternion
import random
import glob
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

        # Sample navigable points
        self.pts = self._sample_points()

        # Bin points based on x and z values, so that
        # we can quickly pool them based on y-filtering
        self.y = self.pts[:, 1]
        (
            self.xz_origin,  # in meters
            self.xz_min,     # in map coordinates
            self.xz_max,     # in map coordinates
            self.xz_size,    # in map coordinates
            self.xz          # in map coordinates
        ) = self._make_map(self.pts[:, [0, 2]])

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
        # Determine map boundaries
        min_ = np.floor(np.min(xz, axis=0) - self.padding).astype(int)
        max_ = np.ceil(np.max(xz, axis=0) + self.padding).astype(int)
        size = np.ceil((max_ - min_ + 1) / self.resolution).astype(int)
        origin = min_ / 100.  # cm to m
        min_ = (min_ / self.resolution).astype(int)
        max_ = min_ + size - 1

        # Recenter points
        xz = (xz / self.resolution).astype(int)
        xz = xz - min_

        return origin, min_, max_, size, xz

    def _get_floor_heights(self):
        floor_heights = []
        hist = np.histogram(
            np.asarray(self.y),
            bins=np.arange(
                self.y.min(), self.y.max() + self.floor_thr, self.floor_thr
            )
        )
        for i in range(len(hist[0])):
            if hist[0][i] > self.num_sampled_points // 5:
                floor_heights.append((hist[1][i] + hist[1][i + 1]) / 2)
        return floor_heights

    def _get_floor_navigable_map(self, y):
        ids = np.logical_and(self.y > y - self.floor_thr,
                             self.y < y + self.floor_thr)
        map = np.zeros((self.xz_size[0], self.xz_size[1]), dtype=int)
        np.add.at(map, (self.xz[ids, 0], self.xz[ids, 1]), 1)
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
                bbox[0] - self.xz_min[0],
                bbox[1] - self.xz_min[1],
                bbox[2] - self.xz_min[0],
                bbox[3] - self.xz_min[1]
            ]

            if -0.25 < obj.aabb.center[1] - y / 100. < 1.5:
                sem_map[category_id + 1, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1.
                sem_map[0, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0.

        return sem_map

    def _get_floor_semantic_map_from_first_person(self, y, num_frames=2):
        ids = np.logical_and(self.y > y - self.floor_thr,
                             self.y < y + self.floor_thr)
        positions = self.pts[ids] / 100.  # cm to m

        idxs = random.sample(range(len(positions)), num_frames)
        positions = positions[idxs]

        # TODO Batch positions

        sequence_length = positions.shape[0]
        yaws = np.random.random(sequence_length)
        rotations = quaternion.from_euler_angles(0., yaws, 0.)
        seq_obs = [self.sim.get_observations_at(positions[t], rotations[t])
                   for t in range(sequence_length)]
        for t in range(sequence_length):
            seq_obs[t]["gps"] = positions[t, [0, 2]]
            seq_obs[t]["compass"] = [yaws[t]]

        # Preprocess observations
        (
            seq_obs_preprocessed,
            seq_semantic_frame,
            seq_pose_delta,
            goal_category,
            goal_name
        ) = self.obs_preprocessor.preprocess_sequence(seq_obs)

        seq_dones = torch.tensor([False] * sequence_length)
        seq_update_global = torch.tensor([False] * sequence_length)
        seq_update_global[-1] = True

        # Update map with observations
        (
            seq_map_features,
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

        # TODO Get this working locally (with detectron2 in habitat-sim env?)
        print(self.semantic_map.global_map.shape)


def visualize_sem_map(sem_map):
    def compress_sem_map(sem_map):
        c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
        for i in range(sem_map.shape[0]):
            c_map[sem_map[i] > 0.] = i+1
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
                                 generation_method="annotations_first_person"):
    scene_id = scene_path.split("/")[-1].split(".")[0]

    config, _ = get_config("submission/configs/ddppo_train_challenge_dataset_config.yaml")
    config.defrost()
    if generation_method == "annotations_first_person":
        config.GROUND_TRUTH_SEMANTICS = 1
    task_config = config.TASK_CONFIG
    task_config.SIMULATOR.SCENE = scene_path
    task_config.SIMULATOR.SCENE_DATASET = f"{SCENES_ROOT_PATH}/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    config.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=task_config.SIMULATOR)
    device = torch.device("cpu")  # TODO We can distribute this across GPUs
    floor_maps = HabitatFloorMaps(sim, generation_method, config, device)

    for i, sem_map in enumerate(floor_maps.floor_semantic_maps):
        sem_map_vis = visualize_sem_map(sem_map)
        sem_map_vis.save(f"scenes/{scene_id}_{i}.png", "PNG")


for split in ["val"]:
    # Select scenes with semantic annotations
    scenes = glob.glob(f"{SCENES_ROOT_PATH}/hm3d/{split}/*/*semantic.glb")
    scenes = [scene.replace("semantic.glb", "basis.glb") for scene in scenes]

    # with multiprocessing.Pool(80) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    #     for _ in pool.imap_unordered(generate_scene_ground_truth_maps, scenes):
    #         pbar.update()
    for scene in scenes:
        generate_scene_semantic_maps(scene)
