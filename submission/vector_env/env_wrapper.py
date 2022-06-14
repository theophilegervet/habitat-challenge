import torch
from typing import Tuple

from habitat import Config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from submission.obs_preprocessor.obs_preprocessor import ObsPreprocessor
from submission.planner.planner import Planner
from submission.visualizer.visualizer import Visualizer


class EnvWrapper(Env):

    def __init__(self, config: Config):
        super().__init__(config=config.TASK_CONFIG)

        self.device = (torch.device("cpu") if config.NO_GPU else
                       torch.device(f"cuda:{self.sim.gpu_device}"))
        self.max_steps = config.AGENT.max_steps
        self.planner = Planner(config)
        self.visualizer = Visualizer(config)
        self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)

        self.scene_id = None
        self.episode_id = None
        self.last_semantic_frame = None
        self.last_goal_name = None

    def reset(self) -> Tuple[torch.Tensor, dict]:
        obs = super().reset()
        self.obs_preprocessor.reset()
        self.planner.reset()
        self.visualizer.reset()

        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id
        self._set_vis_dir(self.scene_id, self.episode_id)

        obs_preprocessed, info = self._preprocess_obs(obs)
        return obs_preprocessed, info

    def _preprocess_obs(self, obs: Observations) -> Tuple[torch.Tensor, dict]:
        (
            obs_preprocessed,
            semantic_frame,
            pose_delta,
            goal_category,
            goal_name
        ) = self.obs_preprocessor.preprocess([obs])

        self.last_semantic_frame = semantic_frame[0]
        self.last_goal_name = goal_name[0]

        info = {
            "pose_delta": pose_delta,
            "goal_category": goal_category
        }

        return obs_preprocessed, info

    def plan_and_step(self,
                      planner_inputs: dict,
                      vis_inputs: dict
                      ) -> Tuple[Observations, bool, dict]:
        # 1 - Visualization of previous timestep - now that we have
        #  all necessary components
        vis_inputs["semantic_frame"] = self.last_semantic_frame
        vis_inputs["goal_name"] = self.last_goal_name
        self.visualizer.visualize(**planner_inputs, **vis_inputs)

        # 2 - Planning
        if vis_inputs["timestep"] > self.max_steps:
            action = HabitatSimActions.STOP
        else:
            action = self.planner.plan(**planner_inputs)

        # 3 - Step
        obs = self.step(action)

        # 4 - Preprocess obs - if done, record episode metrics and
        #  reset environment
        done = self.episode_over
        if done:
            done_info = {
                "last_episode_scene_id": self.scene_id,
                "last_episode_id": self.episode_id,
                "last_episode_metrics": self.get_metrics()
            }
            obs_preprocessed, info = self.reset()
            info.update(done_info)

        else:
            obs_preprocessed, info = self._preprocess_obs(obs)

        return obs_preprocessed, done, info

    def _set_vis_dir(self, scene_id: str, episode_id: str):
        """Reset visualization directory."""
        self.planner.set_vis_dir(scene_id, episode_id)
        self.visualizer.set_vis_dir(scene_id, episode_id)
