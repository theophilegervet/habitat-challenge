from typing import Tuple
import torch
from torch import Tensor, IntTensor
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import skimage.morphology
import time

import submission.utils.depth_utils as du
import submission.utils.rotation_utils as ru
import submission.utils.pose_utils as pu
from .common import (
    MapSizeParameters,
    init_map_and_pose_for_env,
    recenter_local_map_and_pose_for_env,
)


class SemanticMapModule(nn.Module):
    """
    This class is responsible for updating the local and global maps and poses
    and generating map features — it is a stateless PyTorch module with no
    trainable parameters.
    """

    def __init__(self, config):
        """
        Arguments decided by the environment:
            frame_height (int): first-person frame height
            frame_width (int): first-person frame width
            camera_height (int): camera sensor height (in metres)
            hfov (int): horizontal field of view (in degrees)
            num_sem_categories (int): number of semantic segmentation
             categories

        Arguments that are important parameters to set:
            map_size_cm (int): global map size (in centimetres)
            map_resolution (int): size of map bins (in centimeters)
            vision_range (int): diameter of the circular region of the
             local map that is visible by the agent located in its center
             (unit is the number of local map cells)
            global_downscaling (int): ratio of global over local map
            du_scale (int): depth unit scale
            cat_pred_threshold (float): number of depth points to be in bin to
             classify it as a certain semantic category
            exp_pred_threshold (float): number of depth points to be in bin to
             consider it as explored
            map_pred_threshold (float): number of depth points to be in bin to
             consider it as obstacle
        """
        super().__init__()

        self.screen_h = config.ENVIRONMENT.frame_height
        self.screen_w = config.ENVIRONMENT.frame_width
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, config.ENVIRONMENT.hfov)
        self.agent_height = config.ENVIRONMENT.camera_height * 100.
        self.num_sem_categories = config.ENVIRONMENT.num_sem_categories

        self.map_size_parameters = MapSizeParameters(config)
        self.resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.global_map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.global_downscaling = config.AGENT.SEMANTIC_MAP.global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        global_map_size = self.global_map_size_cm // self.resolution
        self.global_h, self.global_w = global_map_size, global_map_size
        local_map_size = self.local_map_size_cm // self.resolution
        self.local_h, self.local_w = local_map_size, local_map_size

        self.xy_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.z_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.global_map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.global_downscaling = config.AGENT.SEMANTIC_MAP.global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.vision_range = config.AGENT.SEMANTIC_MAP.vision_range
        self.du_scale = config.AGENT.SEMANTIC_MAP.du_scale
        self.cat_pred_threshold = config.AGENT.SEMANTIC_MAP.cat_pred_threshold
        self.exp_pred_threshold = config.AGENT.SEMANTIC_MAP.exp_pred_threshold
        self.map_pred_threshold = config.AGENT.SEMANTIC_MAP.map_pred_threshold

        # TODO How are min_height and max_height set? Why is min_height negative?
        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2.]

    @torch.no_grad()
    def forward(self,
                seq_obs: Tensor,
                seq_pose_delta: Tensor,
                seq_dones: Tensor,
                seq_update_global: Tensor,
                init_local_map: Tensor,
                init_global_map: Tensor,
                init_local_pose: Tensor,
                init_global_pose: Tensor,
                init_lmb: Tensor,
                init_origins: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        """Update maps and poses with a sequence of observations and generate map
        features at each time step.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) binary flags
             that indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_map_features: sequence of semantic map features of shape
             (batch_size, sequence_length, 8 + num_sem_categories, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        batch_size, sequence_length = seq_obs.shape[:2]
        device, dtype = seq_obs.device, seq_obs.dtype

        map_features_channels = 8 + self.num_sem_categories
        seq_map_features = torch.zeros(
            batch_size,
            sequence_length,
            map_features_channels,
            self.local_h,
            self.local_w,
            device=device,
            dtype=dtype
        )
        seq_local_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_global_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_lmb = torch.zeros(batch_size, sequence_length, 4,
                              device=device, dtype=torch.int32)
        seq_origins = torch.zeros(batch_size, sequence_length, 3, device=device)

        local_map, local_pose = init_local_map.clone(), init_local_pose.clone()
        global_map, global_pose = init_global_map.clone(), init_global_pose.clone()
        lmb, origins = init_lmb.clone(), init_origins.clone()

        for t in range(sequence_length):
            # Reset map and pose for episodes done at time step t
            for e in range(batch_size):
                if seq_dones[e, t]:
                    init_map_and_pose_for_env(
                        e, local_map, global_map, local_pose,
                        global_pose, lmb, origins, self.map_size_parameters
                    )

            local_map, local_pose = self._update_local_map_and_pose(
                seq_obs[:, t], seq_pose_delta[:, t], local_map, local_pose)

            for e in range(batch_size):
                if seq_update_global[e, t]:
                    self._update_global_map_and_pose_for_env(
                        e, local_map, global_map, local_pose, global_pose, lmb, origins)

            seq_local_pose[:, t] = local_pose
            seq_global_pose[:, t] = global_pose
            seq_lmb[:, t] = lmb
            seq_origins[:, t] = origins
            seq_map_features[:, t] = self._get_map_features(local_map, global_map)

        return (
            seq_map_features,
            local_map,
            global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )

    def _update_local_map_and_pose(self,
                                   obs: Tensor,
                                   pose_delta: Tensor,
                                   prev_map: Tensor,
                                   prev_pose: Tensor
                                   ) -> Tuple[Tensor, Tensor]:
        """Update local map and sensor pose given a new observation using parameter-free
        differentiable projective geometry.

        Args:
            obs: current frame containing (rgb, depth, segmentation) of shape
             (batch_size, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: delta in pose since last frame of shape (batch_size, 3)
            prev_map: previous local map of shape
             (batch_size, 4 + num_sem_categories, M, M)
            prev_pose: previous pose of shape (batch_size, 3)

        Returns:
            current_map: current local map updated with current observation
             and location of shape (batch_size, 4 + num_sem_categories, M, M)
            current_pose: current pose updated with pose delta of shape (batch_size, 3)
        """
        t0 = time.time()

        batch_size, obs_channels, h, w = obs.size()
        device, dtype = obs.device, obs.dtype

        depth = obs[:, 3, :, :].float()
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, device)

        voxel_channels = 1 + self.num_sem_categories

        init_grid = torch.zeros(
            batch_size,
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_height - self.min_height,
            device=device,
            dtype=torch.float32
        )
        feat = torch.ones(
            batch_size,
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32
        )
        feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(batch_size, obs_channels - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / self.xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               self.vision_range // 2.) / self.vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (XYZ_cm_std[..., 2] - (self.max_height + self.min_height) // 2.) /
            (self.max_height - self.min_height) * 2.
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3]
        )

        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)

        # TODO What is this 25?
        min_z = int(25 / self.z_resolution - self.min_height)
        max_z = int((self.agent_height + 1) / self.z_resolution - self.min_height)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold

        agent_view = torch.zeros(
            batch_size, obs_channels,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype
        )

        x1 = (self.local_map_size_cm // (self.xy_resolution * 2) -
              self.vision_range // 2)
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = torch.clamp(fp_map_pred, min=0., max=1.)
        agent_view[:, 1:2, y1:y2, x1:x2] = torch.clamp(fp_exp_pred, min=0., max=1.)
        agent_view[:, 4:4 + self.num_sem_categories, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:1 + self.num_sem_categories] / self.cat_pred_threshold,
            min=0., max=1
        )

        current_pose = pu.get_new_pose_batch(prev_pose.clone(), pose_delta)
        st_pose = current_pose.clone().detach()

        st_pose[:, :2] = - (
            (st_pose[:, :2] * 100. / self.xy_resolution -
             self.local_map_size_cm // (self.xy_resolution * 2)) /
            (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = ru.get_grid(st_pose, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps = torch.cat((prev_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        current_map, _ = torch.max(maps[:, :, :4 + self.num_sem_categories], 1)

        t1 = time.time()
        print(f"t1 - t0: {t1 - t0}")

        # Reset current location
        current_map[:, 2, :, :].fill_(0.)
        curr_loc = current_pose[:, :2]
        curr_loc = (curr_loc * 100. / self.xy_resolution).int()
        for e in range(batch_size):
            x, y = curr_loc[e]
            current_map[e, 2:4, y - 2:y + 3, x - 2:x + 3].fill_(1.)

            # Set a disk around the agent to explored
            # TODO Move this to GPU
            # TODO Check that the explored disk fits in the map
            radius = 10
            explored_disk = torch.from_numpy(skimage.morphology.disk(radius))
            current_map[
                e,
                1,
                y - radius: y + radius + 1,
                x - radius: x + radius + 1
            ][explored_disk == 1] = 1

        t2 = time.time()
        print(f"t2 - t1: {t2 - t1}")

        return current_map, current_pose

    def _update_global_map_and_pose_for_env(self,
                                            e: int,
                                            local_map: Tensor,
                                            global_map: Tensor,
                                            local_pose: Tensor,
                                            global_pose: Tensor,
                                            lmb: Tensor,
                                            origins: Tensor):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """
        global_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = local_map[e]
        global_pose[e] = local_pose[e] + origins[e]
        recenter_local_map_and_pose_for_env(
            e, local_map, global_map, local_pose,
            global_pose, lmb, origins, self.map_size_parameters
        )

    def _get_map_features(self, local_map: Tensor, global_map: Tensor) -> Tensor:
        """Get global and local map features.

        Arguments:
            local_map: local map of shape
             (batch_size, 4 + num_sem_categories, M, M)
            global_map: global map of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)

        Returns:
            map_features: semantic map features of shape
             (batch_size, 8 + num_sem_categories, M, M)
        """
        map_features_channels = 8 + self.num_sem_categories

        map_features = torch.zeros(
            local_map.size(0),
            map_features_channels,
            self.local_h,
            self.local_w,
            device=local_map.device,
            dtype=local_map.dtype
        )

        # Local obstacles, explored area, and current and past position
        map_features[:, 0:4, :, :] = local_map[:, 0:4, :, :]
        # Global obstacles, explored area, and current and past position
        map_features[:, 4:8, :, :] = nn.MaxPool2d(self.global_downscaling)(
            global_map[:, 0:4, :, :])
        # Local semantic categories
        map_features[:, 8:, :, :] = local_map[:, 4:, :, :]

        return map_features.detach()
