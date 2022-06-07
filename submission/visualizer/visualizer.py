import os
import shutil
import numpy as np
import cv2
from PIL import Image
import skimage.morphology

from habitat import Config

import submission.utils.visualization_utils as vu
import submission.utils.pose_utils as pu
from submission.utils.constants import map_color_palette


class Visualizer:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, config: Config):
        self.show_images = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES

        self.color_palette = [int(x * 255.) for x in map_color_palette]
        self.legend = cv2.imread("submission/visualizer/legend.png")

        self.images_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        shutil.rmtree(self.images_dir, ignore_errors=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.num_sem_categories = config.ENVIRONMENT.num_sem_categories
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        self.map_shape = (
            config.AGENT.SEMANTIC_MAP.map_size_cm // self.map_resolution,
            config.AGENT.SEMANTIC_MAP.map_size_cm // self.map_resolution
        )

        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None

    def reset(self):
        self.image_vis = None
        self.visited_map_vis = np.zeros(self.map_shape)
        self.last_xy = None

    def visualize(self,
                  sensor_pose: np.ndarray,
                  obstacle_map: np.ndarray,
                  goal_map: np.ndarray,
                  explored_map: np.ndarray,
                  semantic_map: np.ndarray,
                  semantic_frame: np.ndarray,
                  goal_name: str,
                  timestep: int):
        """Visualize frame input and semantic map.

        Args:
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gy1, gy2, gx1, gy2)
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            explored_map: (M, M) binary local explored map prediction
            semantic_map: (M, M) local semantic map predictions
            semantic_frame: semantic frame visualization
            goal_name: semantic goal category
            timestep: time stamp within the episode
        """
        if self.image_vis is None:
            self.image_vis = vu.init_vis_image(goal_name, self.legend)

        curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
        gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)

        # Update visited map with last visited area
        if self.last_xy is not None:
            last_x, last_y = self.last_xy
            last_pose = [int(last_y * 100. / self.map_resolution - gy1),
                         int(last_x * 100. / self.map_resolution - gx1)]
            last_pose = pu.threshold_poses(last_pose, obstacle_map.shape)
            curr_pose = [int(curr_y * 100. / self.map_resolution - gy1),
                         int(curr_x * 100. / self.map_resolution - gx1)]
            curr_pose = pu.threshold_poses(curr_pose, obstacle_map.shape)
            self.visited_map_vis[gy1:gy2, gx1:gx2] = vu.draw_line(
                last_pose, curr_pose, self.visited_map_vis[gy1:gy2, gx1:gx2])
        self.last_xy = (curr_x, curr_y)

        semantic_map += 5

        # Obstacles, explored, and visited areas
        no_category_mask = semantic_map == 5 + self.num_sem_categories - 1
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = self.visited_map_vis[gy1:gy2, gx1:gx2] == 1
        semantic_map[no_category_mask] = 0
        semantic_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
        semantic_map[np.logical_and(no_category_mask, explored_mask)] = 2
        semantic_map[visited_mask] = 3

        # Goal
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(goal_map, selem) != True
        goal_mask = goal_mat == 1
        semantic_map[goal_mask] = 4

        # Semantic categories
        semantic_map_vis = Image.new("P", (semantic_map.shape[1], semantic_map.shape[0]))
        semantic_map_vis.putpalette(self.color_palette)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]
        semantic_map_vis = cv2.resize(semantic_map_vis, (480, 480),
                                      interpolation=cv2.INTER_NEAREST)
        self.image_vis[50:530, 670:1150] = semantic_map_vis

        # First-person semantic frame
        self.image_vis[50:530, 15:655] = cv2.resize(semantic_frame, (640, 480))

        # Agent arrow
        pos = (
            (curr_x * 100. / self.map_resolution - gx1)
            * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100. / self.map_resolution + gy1)
            * 480 / obstacle_map.shape[1],
            np.deg2rad(-curr_o)
        )
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = self.color_palette[9:12][::-1]
        cv2.drawContours(self.image_vis, [agent_arrow], 0, color, -1)

        if self.show_images:
            raise NotImplementedError

        if self.print_images:
            cv2.imwrite(os.path.join(self.images_dir, f"{timestep}.png"),
                        self.image_vis)
