from typing import Tuple

from habitat import Config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from submission.planner.planner import Planner
from submission.visualizer.visualizer import Visualizer


class EnvWrapper(Env):

    def __init__(self, config: Config):
        super().__init__(config=config.TASK_CONFIG)

        self.max_steps = config.AGENT.max_steps
        self.planner = Planner(config)
        self.visualizer = Visualizer(config)

        self.scene_id = None
        self.episode_id = None

    def reset(self):
        obs = super().reset()
        self.planner.reset()
        self.visualizer.reset()

        self.scene_id = self.current_episode.scene_id.split("/")[-1].split(".")[0]
        self.episode_id = self.current_episode.episode_id
        self._set_vis_dir(self.scene_id, self.episode_id)

        return obs

    def plan_and_step(self,
                      planner_inputs: dict,
                      vis_inputs: dict
                      ) -> Tuple[Observations, bool, dict]:
        # 1 - Planning
        if vis_inputs["timestep"] > self.max_steps:
            action = HabitatSimActions.STOP
        else:
            action = self.planner.plan(**planner_inputs)

        # 2 - Visualization
        self.visualizer.visualize(**planner_inputs, **vis_inputs)

        # 3 - Step
        obs = self.step(action)

        # 4 - Prepare done and info
        done = self.episode_over
        info = {"last_action": action}

        # 5 - If done, record episode metrics and reset environment
        if done:
            info = {
                "last_action": None,
                "last_episode_scene_id": self.scene_id,
                "last_episode_id": self.episode_id,
                "last_episode_metrics": self.get_metrics()
            }
            obs = self.reset()

        return obs, done, info

    def _set_vis_dir(self, scene_id: str, episode_id: str):
        """Reset visualization directory."""
        self.planner.set_vis_dir(scene_id, episode_id)
        self.visualizer.set_vis_dir(scene_id, episode_id)
