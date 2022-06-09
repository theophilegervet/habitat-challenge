import os
import shutil
import cv2
import math
import numpy as np
from typing import Tuple, List
import skimage.morphology
import time

from habitat import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import submission.utils.pose_utils as pu
from .fmm_planner import FMMPlanner


class Planner:
    """
    This class translates planner inputs into a low-level action — it is
    conceptually part of the agent, but it's more efficient to include it
    in the environment when training with vectorized environments.
    """

    def __init__(self, config: Config):
        """
        Arguments decided by the environment:
            frame_height (int): first-person frame height
            frame_width (int): first-person frame width
            num_sem_categories (int): number of semantic segmentation
             categories
            turn_angle (float): agent turn angle (in degrees)

        Arguments that are important parameters to set:
            map_size_cm (int): map size (in centimetres)
            map_resolution (int): size of map bins (in centimeters)
            collision_threshold (float): distance under which we consider
             there's a collision
        """
        self.visualize = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES
        self.default_vis_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.map_shape = (self.map_size_cm // self.map_resolution,
                          self.map_size_cm // self.map_resolution)
        self.num_sem_categories = config.ENVIRONMENT.num_sem_categories
        self.frame_width = config.ENVIRONMENT.frame_width
        self.frame_height = config.ENVIRONMENT.frame_height
        self.obs_shape = (3 + 1 + self.num_sem_categories,
                          self.frame_height, self.frame_width)
        self.stop_distance = config.ENVIRONMENT.success_distance
        self.turn_angle = config.ENVIRONMENT.turn_angle
        self.collision_threshold = config.AGENT.PLANNER.collision_threshold
        if config.AGENT.PLANNER.denoise_selem_radius > 0:
            self.denoise_selem = skimage.morphology.disk(
                config.AGENT.PLANNER.denoise_selem_radius)
        else:
            self.denoise_selem = None
        self.dilation_selem = skimage.morphology.disk(
            config.AGENT.PLANNER.dilation_selem_radius)

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = None

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [self.map_size_cm / 100. / 2.,
                          self.map_size_cm / 100. / 2., 0.]
        self.last_action = None
        self.timestep = 1

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.vis_dir = os.path.join(
            self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def plan(self,
             obstacle_map: np.ndarray,
             sensor_pose: np.ndarray,
             goal_map: np.ndarray) -> int:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            goal_map: (M, M) binary array denoting goal location

        Returns:
            action: low-level action
        """
        self.last_pose = self.curr_pose
        obstacle_map = np.rint(obstacle_map)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        start = [int(start_y * 100. / self.map_resolution - gx1),
                 int(start_x * 100. / self.map_resolution - gy1)]
        start = pu.threshold_poses(start, obstacle_map.shape)

        # If we're close enough to the closest goal, stop
        # TODO Compute distance in meters instead of map cells
        goal_locations = np.argwhere(goal_map == 1)
        distances = np.linalg.norm(goal_locations - start, axis=1)
        if distances.min() < 12.:
            return HabitatSimActions.STOP

        self.curr_pose = [start_x, start_y, start_o]
        self.visited_map[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                           start[1] - 0:start[1] + 1] = 1

        if self.last_action == HabitatSimActions.MOVE_FORWARD:
            self._check_collision()

        # High-level goal -> short-term goal
        t0 = time.time()
        short_term_goal, stop = self._get_short_term_goal(
            obstacle_map, np.copy(goal_map), start, planning_window)
        t1 = time.time()
        print(f"Planning get_short_term_goal() time: {t1 - t0}")

        # Short-term goal -> deterministic local policy
        if stop:
            action = HabitatSimActions.STOP
        else:
            stg_x, stg_y = short_term_goal
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = start_o % 360.
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.turn_angle / 2.:
                action = HabitatSimActions.TURN_RIGHT
            elif relative_angle < -self.turn_angle / 2.:
                action = HabitatSimActions.TURN_LEFT
            else:
                action = HabitatSimActions.MOVE_FORWARD

        self.last_action = action
        return action

    def _get_short_term_goal(self,
                             obstacle_map: np.ndarray,
                             goal_map: np.ndarray,
                             start: List[int],
                             planning_window: List[int],
                             ) -> Tuple[Tuple[int, int], bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            stop: binary flag to indicate we've reached the goal
        """
        gx1, gx2, gy1, gy2 = planning_window
        x1, y1, = 0, 0
        x2, y2 = obstacle_map.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        obstacles = obstacle_map[x1:x2, y1:y2]

        # Remove noise with standard morphological transformation
        # (closing -> opening)
        if self.denoise_selem is not None:
            denoised_obstacles = cv2.morphologyEx(
                obstacles,
                cv2.MORPH_CLOSE,
                self.denoise_selem
            )
            denoised_obstacles = cv2.morphologyEx(
                denoised_obstacles,
                cv2.MORPH_OPEN,
                self.denoise_selem
            )
        else:
            denoised_obstacles = obstacles

        # Increase the size of obstacles
        dilated_obstacles = cv2.dilate(
            denoised_obstacles,
            self.dilation_selem,
            iterations=1
        )

        # if self.visualize:
        #     r, c = obstacles.shape
        #     morphological_vis = np.zeros((r, c * 3))
        #     morphological_vis[:, :c] = obstacles
        #     morphological_vis[:, c:2 * c] = denoised_obstacles
        #     morphological_vis[:, 2 * c:] = dilated_obstacles
        #     cv2.imshow("Planner Morphological Transformations", morphological_vis)
        #     cv2.waitKey(1)

        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1
        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)

        planner = FMMPlanner(
            traversible,
            self.stop_distance,
            vis_dir=self.vis_dir,
            visualize=self.visualize,
            print_images=self.print_images
        )

        selem = skimage.morphology.disk(10)
        goal_map = skimage.morphology.binary_dilation(goal_map, selem) != True
        goal_map = 1 - goal_map * 1.
        planner.set_multi_goal(goal_map, self.timestep)
        self.timestep += 1

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        short_term_goal = int(stg_x), int(stg_y)
        return short_term_goal, stop

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * \
                         ((i + buf) * np.cos(np.deg2rad(t1))
                          + (j - width // 2) * np.sin(np.deg2rad(t1)))
                    wy = y1 + 0.05 * \
                         ((i + buf) * np.sin(np.deg2rad(t1))
                          - (j - width // 2) * np.cos(np.deg2rad(t1)))
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), \
                           int(c * 100 / self.map_resolution)
                    [r, c] = pu.threshold_poses([r, c],
                                                self.collision_map.shape)
                    self.collision_map[r, c] = 1
