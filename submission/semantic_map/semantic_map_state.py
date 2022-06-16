import torch
import numpy as np

from .common import MapSizeParameters, init_map_and_pose_for_env


class SemanticMapState:
    """
    This class holds the global and local map and sensor pose, as well as the
    agent's current goal.
    """

    def __init__(self, config, rank):
        self.device = (torch.device("cpu") if config.NO_GPU else
                       torch.device(f"cuda:{config.AGENT_GPU_IDS[rank]}"))
        self.precision = torch.float16 if config.MIXED_PRECISION else torch.float32
        self.num_environments = config.NUM_ENVIRONMENTS
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

        # Map consists of multiple channels containing the following:
        # 0: Obstacle Map
        # 1: Explored Area
        # 2: Current Agent Location
        # 3: Past Agent Locations
        # 4, 5, 6, .., num_sem_categories + 3: Semantic Categories
        num_channels = self.num_sem_categories + 4

        self.global_map = torch.zeros(
            self.num_environments, num_channels, self.global_h, self.global_w,
            device=self.device, dtype=self.precision)
        self.local_map = torch.zeros(
            self.num_environments, num_channels, self.local_h, self.local_w,
            device=self.device, dtype=self.precision)

        # Global and local (x, y, o) sensor pose
        self.global_pose = torch.zeros(self.num_environments, 3, device=self.device)
        self.local_pose = torch.zeros(self.num_environments, 3, device=self.device)

        # Origin of local map (3rd dimension stays 0)
        self.origins = torch.zeros(self.num_environments, 3, device=self.device)

        # Local map boundaries
        self.lmb = torch.zeros(self.num_environments, 4, dtype=torch.int32, device=self.device)

        # Binary map encoding agent high-level goal
        self.goal_map = np.zeros((self.num_environments, self.local_h, self.local_w))

    def init_map_and_pose(self):
        """Initialize global and local map and sensor pose variables."""
        for e in range(self.num_environments):
            self.init_map_and_pose_for_env(e)

    def init_map_and_pose_for_env(self, e: int):
        """Initialize global and local map and sensor pose variables for
        a specific environment.
        """
        init_map_and_pose_for_env(
            e, self.local_map, self.global_map, self.local_pose, self.global_pose,
            self.lmb, self.origins, self.map_size_parameters)

    def update_global_goal_for_env(self, e: int, goal_map: np.ndarray):
        """Update global goal for a specific environment with the goal action chosen
        by the policy.

        Arguments:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        self.goal_map[e] = goal_map

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_obstacle_map(self, e) -> np.ndarray:
        """Get local obstacle map for an environment."""
        return np.copy(self.local_map[e, 0, :, :].cpu().float().numpy())

    def get_explored_map(self, e) -> np.ndarray:
        """Get local explored map for an environment."""
        return np.copy(self.local_map[e, 1, :, :].cpu().float().numpy())

    def get_semantic_map(self, e) -> np.ndarray:
        """Get local map of semantic categories for an environment."""
        semantic_map = np.copy(self.local_map[e].cpu().float().numpy())
        semantic_map[3 + self.num_sem_categories, :, :] = 1e-5  # Last category is unlabeled
        semantic_map = semantic_map[4:4 + self.num_sem_categories, :, :].argmax(0)
        return semantic_map

    def get_planner_pose_inputs(self, e) -> np.ndarray:
        """Get local planner pose inputs for an environment.

        Returns:
            planner_pose_inputs with 7 dimensions:
             1-3: Global pose
             4-7: Local map boundaries
        """
        return torch.cat([
            self.local_pose[e] + self.origins[e],
            self.lmb[e]
        ]).cpu().float().numpy()

    def get_goal_map(self, e) -> np.ndarray:
        """Get binary goal map encoding current global goal for an
        environment."""
        return self.goal_map[e]
