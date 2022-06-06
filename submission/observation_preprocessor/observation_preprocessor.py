from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from habitat import Config
from habitat.core.simulator import Observations


class ObservationPreprocessor:

    def __init__(self,
                 config: Config,
                 device: torch.device,
                 ):
        self.device = device
        self.precision = torch.float16 if config.MIXED_PRECISION_AGENT else torch.float32

        self.segmentation = # TODO

        self.one_hot_encoding = torch.eye(config.ENVIRONMENT.num_sem_categories,
                                          device=self.device, dtype=self.precision)
        self.frame_height = config.ENVIRONMENT.frame_height
        self.frame_width = config.ENVIRONMENT.frame_width
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth

    def preprocess(self, obs: List[Observations]) -> Tuple[Tensor, Tensor]:
        obs_preprocessed = self.preprocess_frame(obs)
        pose_delta = self.preprocess_pose(obs)
        return obs_preprocessed, pose_delta

    def preprocess_frame(self, obs: List[Observations]) -> Tuple[torch.Tensor, np.ndarray]:
        def preprocess_depth(depth):
            zero_mask = depth == 0.
            col_max = depth.max(axis=1, keepdims=True).values
            depth += zero_mask * col_max
            return depth

        def downscale(rgb, depth, semantic):
            h_downscaling = env_frame_height // self.frame_height
            w_downscaling = env_frame_width // self.frame_width
            assert h_downscaling == w_downscaling
            assert type(h_downscaling) == int
            if h_downscaling == 1:
                return rgb, depth, semantic
            else:
                rgb = F.interpolate(rgb, scale_factor=1. / h_downscaling,
                                    mode='bilinear')
                depth = F.interpolate(depth,
                                      scale_factor=1. / h_downscaling,
                                      mode='bilinear')
                semantic = F.interpolate(semantic,
                                         scale_factor=1. / h_downscaling,
                                         mode='nearest')
                return rgb, depth, semantic

        env_frame_height, env_frame_width = obs[0]["rgb"].shape[:2]

        rgb = torch.from_numpy(np.stack([ob["rgb"] for ob in obs])).to(
            self.precision).to(self.device)
        depth = torch.from_numpy(np.stack([ob["depth"] for ob in obs])).to(
            self.precision).to(self.device)

        depth = preprocess_depth(depth)

        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        semantic, semantic_vis = self.segmentation(rgb, depth)

        semantic = self.one_hot_encoding[semantic]
        semantic = semantic.permute(0, 3, 1, 2)

        depth = self.min_depth * 100. + depth * self.max_depth * 100.
        rgb, depth, semantic = downscale(rgb, depth, semantic)
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=1)

        return obs_preprocessed, semantic_vis

    def preprocess_pose(self, obs: List[Observations]):
        # TODO
        pass
