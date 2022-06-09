from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from PIL import Image

from habitat import Config
from habitat.core.simulator import Observations

import submission.utils.pose_utils as pu
from submission.utils.constants import (
    frame_color_palette,
    goal_categories,
    goal_categories_mapping
)
from .mask_rcnn import MaskRCNN


class ObsPreprocessor:
    """
    This class preprocesses observations - it can either be integrated in the
    agent or an environment.
    """

    def __init__(self,
                 config: Config,
                 num_environments: int,
                 device: torch.device):
        self.num_environments = num_environments
        self.device = device
        self.precision = torch.float16 if config.MIXED_PRECISION_AGENT else torch.float32
        self.num_sem_categories = config.ENVIRONMENT.num_sem_categories
        self.frame_height = config.ENVIRONMENT.frame_height
        self.frame_width = config.ENVIRONMENT.frame_width
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth

        self.segmentation = MaskRCNN(
            sem_pred_prob_thr=0.9,
            sem_gpu_id=(-1 if device == torch.device("cpu") else device.index),
            visualize=True
        )

        self.instance_id_to_category_id = None
        self.one_hot_encoding = torch.eye(
            self.num_sem_categories, device=self.device, dtype=self.precision)
        self.color_palette = [int(x * 255.) for x in frame_color_palette]

    def set_instance_id_to_category_id(self, instance_id_to_category_id):
        self.instance_id_to_category_id = instance_id_to_category_id.to(self.device)

    def preprocess(self,
                   obs: List[Observations],
                   last_poses: List[np.ndarray]
                   ) -> Tuple[Tensor, np.ndarray, List[np.ndarray],
                              Tensor, Tensor, List[str]]:
        """Preprocess observation."""
        obs_preprocessed, semantic_frame = self.preprocess_frame(obs)
        pose_delta, curr_poses = self.preprocess_pose(obs, last_poses)
        goal, goal_name = self.preprocess_goal(obs)
        return (
            obs_preprocessed,
            semantic_frame,
            curr_poses,
            pose_delta,
            goal,
            goal_name
        )

    def preprocess_goal(self, obs: List[Observations]) -> Tuple[Tensor, List[str]]:
        goal = torch.tensor([
            goal_categories_mapping[ob["objectgoal"][0]] for ob in obs])
        goal_name = [goal_categories[ob["objectgoal"][0]] for ob in obs]
        return goal, goal_name

    def preprocess_frame(self,
                         obs: List[Observations]
                         ) -> Tuple[Tensor, np.ndarray]:
        """Preprocess frame information in the observation."""
        def preprocess_depth(depth):
            zero_mask = depth == 0.
            col_max = depth.max(axis=1, keepdims=True).values
            depth += zero_mask * col_max
            depth = self.min_depth * 100. + depth * self.max_depth * 100.
            return depth

        def downscale(rgb, depth, semantic):
            h_downscaling = env_frame_height // self.frame_height
            w_downscaling = env_frame_width // self.frame_width
            assert h_downscaling == w_downscaling
            assert type(h_downscaling) == int
            if h_downscaling == 1:
                return rgb, depth, semantic
            else:
                rgb = F.interpolate(
                    rgb, scale_factor=1. / h_downscaling, mode='bilinear')
                depth = F.interpolate(
                    depth, scale_factor=1. / h_downscaling, mode='bilinear')
                semantic = F.interpolate(
                    semantic, scale_factor=1. / h_downscaling, mode='nearest')
                return rgb, depth, semantic

        env_frame_height, env_frame_width = obs[0]["rgb"].shape[:2]

        rgb = torch.from_numpy(np.stack([ob["rgb"] for ob in obs])).to(
            self.precision).to(self.device)
        depth = torch.from_numpy(np.stack([ob["depth"] for ob in obs])).to(
            self.precision).to(self.device)

        depth = preprocess_depth(depth)

        # TODO Handle more than a single frame
        if "semantic" in obs[0] and self.instance_id_to_category_id is not None:
            # Ground-truth semantic segmentation
            assert "semantic" in obs[0]
            semantic = torch.from_numpy(
                np.stack([ob["semantic"] for ob in obs]
            ).squeeze(-1).astype(np.int64)).to(self.device)
            semantic = self.instance_id_to_category_id[semantic]
            semantic = self.one_hot_encoding[semantic]
            semantic_vis = self._get_semantic_frame_vis(
                rgb[0].cpu().numpy(), semantic[0].cpu().numpy())
            semantic_vis = np.expand_dims(semantic_vis, 0)

        else:
            # Predicted semantic segmentation
            # TODO Avoid conversion to numpy
            # TODO Make sure we're sending data in the right format (BGR vs RGB)
            semantic, semantic_vis = self.segmentation.get_prediction(
                rgb[0].cpu().numpy(), depth[0].cpu().squeeze(-1).numpy())
            semantic = torch.from_numpy(semantic).unsqueeze(0).long().to(self.device)
            semantic_vis = np.expand_dims(semantic_vis, 0)

        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
        semantic = semantic.permute(0, 3, 1, 2)

        rgb, depth, semantic = downscale(rgb, depth, semantic)
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=1)

        return obs_preprocessed, semantic_vis

    def _get_semantic_frame_vis(self, rgb: np.ndarray, semantics: np.ndarray):
        """Visualize first-person semantic segmentation frame."""
        width, height = semantics.shape[:2]
        vis_content = semantics
        vis_content[:, :, -1] = 1e-5
        vis_content = vis_content.argmax(-1)
        vis = Image.new("P", (height, width))
        vis.putpalette(self.color_palette)
        vis.putdata(vis_content.flatten().astype(np.uint8))
        vis = vis.convert("RGB")
        vis = np.array(vis)
        vis = np.where(vis != 255, vis, rgb)
        vis = vis[:, :, ::-1]
        return vis

    def preprocess_pose(self,
                        obs: List[Observations],
                        last_poses: List[np.ndarray]
                        ) -> Tuple[Tensor, List[np.ndarray]]:
        """Preprocess sensor pose information in the observation."""
        curr_poses = []
        pose_deltas = []

        for e in range(self.num_environments):
            curr_pose = np.array([
                obs[e]["gps"][0],
                -obs[e]["gps"][1],
                obs[e]["compass"][0]
            ])
            pose_delta = pu.get_rel_pose_change(curr_pose, last_poses[e])
            curr_poses.append(curr_pose)
            pose_deltas.append(pose_delta)

        return torch.tensor(pose_deltas), curr_poses
