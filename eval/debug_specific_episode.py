import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat.core.env import Env
from habitat.core.simulator import Observations

from submission.utils.config_utils import get_config
from submission.agent import Agent
from submission.utils.constants import mp3d_categories_mapping, hm3d_to_mp3d


def reset_to_episode(env: Env,
                     scene_id: str,
                     episode_id: str) -> Observations:
    """
    Adapted from:
    https://github.com/facebookresearch/habitat-lab/blob/main/habitat/core/env.py
    """
    env._reset_stats()

    episode = [e for e in env.episodes
               if e.episode_id == episode_id and
               e.scene_id.endswith(f"{scene_id}.basis.glb")][0]
    env._current_episode = episode

    env._episode_from_iter_on_reset = True
    env._episode_force_changed = False

    env.reconfigure(env._config)

    observations = env.task.reset(episode=env.current_episode)
    env._task.measurements.reset_measures(
        episode=env.current_episode,
        task=env.task,
        observations=observations,
    )
    return observations


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/eval_hm3d_config.yaml")
    config.defrost()
    config.EXP_NAME = "debug_specific_episode"
    config.TASK_CONFIG.DATASET.SPLIT = "val"
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.freeze()

    agent = Agent(config=config, rank=0, ddp=False)
    env = Env(config=config.TASK_CONFIG)

    scene_id = "ziup5kvtCCR"
    episode_id = "1"

    obs = reset_to_episode(env, scene_id, episode_id)
    agent.reset()
    agent.set_vis_dir(scene_id=scene_id, episode_id=episode_id)
    if config.GROUND_TRUTH_SEMANTICS:
        agent.obs_preprocessor.set_instance_id_to_category_id(
            torch.tensor([
                mp3d_categories_mapping.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip()),
                    config.ENVIRONMENT.num_sem_categories - 1
                )
                for obj in env.sim.semantic_annotations().objects
            ])
        )

    t = 0
    while not env.episode_over:
        t += 1
        print(t)
        action = agent.act(obs)
        obs = env.step(action)

    print(env.get_metrics())
