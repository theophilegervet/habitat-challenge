import torch
import json
import os
from typing import Tuple, Optional, List

from habitat import Config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.dataset import EpisodeIterator

from submission.obs_preprocessor.obs_preprocessor import ObsPreprocessor
from submission.planner.planner import Planner
from submission.visualizer.visualizer import Visualizer
from submission.utils.constants import (
    challenge_goal_name_to_goal_name,
    mp3d_categories_mapping,
    hm3d_to_mp3d
)


class EvalEnvWrapper(Env):
    """
    This environment wrapper is used for evaluation. It contains stepping
    the underlying environment, preprocessing observations, and planning
    given a high-level goal predicted by the policy. It is complemented by
    a semantic map state, update, and high-level goal policy in the agent.
    """

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
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.device = (torch.device("cpu") if config.NO_GPU else
                       torch.device(f"cuda:{self.sim.gpu_device}"))
        self.max_steps = config.AGENT.max_steps
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.forced_episode_ids = episode_ids if episode_ids else []
        self.episode_idx = 0

        # Keep only episodes with a goal on the same floor as the
        #  starting position
        if config.EVAL_VECTORIZED.goal_on_same_floor:
            new_episodes = []
            for episode in self._dataset.episodes:
                scene_dir = "/".join(episode.scene_id.split("/")[:-1])
                map_dir = scene_dir + "/floor_semantic_maps_annotations_top_down"
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]
                with open(f"{map_dir}/{scene_id}_info.json", "r") as f:
                    scene_info = json.load(f)
                start_on_first_floor = abs(episode.start_position[1] * 100. - scene_info["floor_heights_cm"][0]) < 50
                goal_on_same_floor = len([
                    goal for goal in episode.goals
                    if episode.start_position[1] - 0.25 < goal.position[1] < episode.start_position[1] + 1.5
                ]) > 0
                if start_on_first_floor and goal_on_same_floor:
                    new_episodes.append(episode)

            # TODO - Keep at least one episode to avoid environment crashing,
            #  there's probably a cleaner way to do this
            if len(new_episodes) == 0:
                new_episodes = [self._dataset.episodes[0]]

            print(f"From {len(self._dataset.episodes)} total episodes for this "
                  f"environment to {len(new_episodes)} on the same floor")
            self._dataset.episodes = new_episodes
            self.episode_iterator = EpisodeIterator(
                new_episodes,
                shuffle=False, group_by_scene=False,
            )
            self._current_episode = None

        # Put episodes with specified object category first
        forced_category = config.EVAL_VECTORIZED.specific_category
        num_ep = config.EVAL_VECTORIZED.num_episodes_per_env
        if forced_category:
            episodes_with_category = []
            other_episodes = []
            idx = 0
            while len(episodes_with_category) < num_ep and idx < len(self.episodes):
                ep = self.episodes[idx]
                cat = challenge_goal_name_to_goal_name[ep.object_category]
                if cat == forced_category:
                    episodes_with_category.append(ep)
                else:
                    other_episodes.append(ep)
                idx += 1
            new_episode_order = [
                *episodes_with_category,
                *other_episodes,
                *self.episodes[idx:]
            ]
            self._dataset.episodes = new_episode_order
            self.episode_iterator = EpisodeIterator(
                new_episode_order,
                shuffle=False, group_by_scene=False,
            )
            self._current_episode = None

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
        self.episode_panorama_start_steps = self.panorama_start_steps

        self.obs_preprocessor.reset()
        self.planner.reset()
        self.visualizer.reset()

        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id
        self._set_vis_dir(self.scene_id, self.episode_id)
        if (len(self.forced_episode_ids) > 0 and
                self.episode_id not in self.forced_episode_ids):
            self._disable_print_images()

        if self.ground_truth_semantics:
            self.obs_preprocessor.set_instance_id_to_category_id(
                torch.tensor([
                    mp3d_categories_mapping.get(
                        hm3d_to_mp3d.get(obj.category.name().lower().strip()),
                        self.obs_preprocessor.num_sem_categories - 1
                    )
                    for obj in self.sim.semantic_annotations().objects
                ])
            )

        obs_preprocessed, info = self._preprocess_obs(obs)

        return obs_preprocessed, info

    def _reset_to_episode(self, episode_id: str) -> Observations:
        """
        Reset the environment to a specific episode ID

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
        if planner_inputs["found_goal"] or planner_inputs["found_hint"]:
            self.episode_panorama_start_steps = 0
        if vis_inputs["timestep"] < self.episode_panorama_start_steps:
            action = HabitatSimActions.TURN_RIGHT
        elif vis_inputs["timestep"] > self.max_steps:
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
