import torch
from typing import Tuple, Optional, List

from habitat import Config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from submission.obs_preprocessor.obs_preprocessor import ObsPreprocessor
from submission.planner.planner import Planner
from submission.visualizer.visualizer import Visualizer


class EnvWrapper(Env):

    def __init__(self,
                 config: Config,
                 episode_ids: Optional[List[str]] = None):
        """
        Arguments:
            episode_ids: if specified, force the environment to iterate
             through these episodes before others - this is useful to
             debug behavior on specific episodes
        """
        super().__init__(config=config.TASK_CONFIG)

        self.device = (torch.device("cpu") if config.NO_GPU else
                       torch.device(f"cuda:{self.sim.gpu_device}"))
        self.max_steps = config.AGENT.max_steps

        self.forced_episode_ids = episode_ids if episode_ids else []
        self.episode_idx = 0

        self.planner = Planner(config)
        self.visualizer = Visualizer(config)
        self.obs_preprocessor = ObsPreprocessor(config, 1, self.device)

        self.scene_id = None
        self.episode_id = None
        self.last_semantic_frame = None
        self.last_goal_name = None

    def reset(self) -> Tuple[torch.Tensor, dict]:
        if self.episode_idx < len(self.forced_episode_ids):
            obs = self._reset_to_episode(
                self.forced_episode_ids[self.episode_idx])
        else:
            obs = super().reset()

        self.episode_idx += 1

        self.obs_preprocessor.reset()
        self.planner.reset()
        self.visualizer.reset()

        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id
        self._set_vis_dir(self.scene_id, self.episode_id)
        if (len(self.forced_episode_ids) > 0 and
                self.episode_id not in self.forced_episode_ids):
            self._disable_print_images()

        obs_preprocessed, info = self._preprocess_obs(obs)
        return obs_preprocessed, info

    def _reset_to_episode(self, episode_id: str) -> Observations:
        """
        Adapted from:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/core/env.py
        """
        self._reset_stats()

        episode = [e for e in self.episodes if e.episode_id == episode_id][0]
        self._current_episode = episode

        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )
        return observations

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
                "last_goal_name": self.last_goal_name,
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

    def _disable_print_images(self):
        self.planner.disable_print_images()
        self.visualizer.disable_print_images()
