import torch
import pprint
from habitat.core.env import Env

from submission.utils.config_utils import get_config
from submission.utils.constants import hm3d_categories_mapping
from submission.agent import Agent


def main():
    config, config_str = get_config("submission/configs/config.yaml")

    agent = Agent(config=config, rank=0, ddp=False)
    env = Env(config=config.TASK_CONFIG)

    episode_metrics = {}

    for ep in range(len(env.episodes)):
        obs = env.reset()
        agent.reset()

        scene_id = env.current_episode.scene_id.split("/")[-1].split(".")[0]
        episode_id = env.current_episode.episode_id
        agent.set_vis_dir(scene_id=scene_id, episode_id=episode_id)

        # Set mapping to convert instance segmentation to semantic segmentation
        # when using ground-truth semantics
        # agent.obs_preprocessor.set_instance_id_to_category_id(torch.tensor([
        #     hm3d_categories_mapping.get(
        #         obj.category.index(),
        #         config.ENVIRONMENT.num_sem_categories - 1
        #     )
        #     for obj in env.sim.semantic_annotations().objects
        # ]))

        while not env.episode_over:
            action = agent.act(obs)
            obs = env.step(action)

        episode_metrics[f"{scene_id}_{episode_id}"] = env.get_metrics()

    aggregated_metrics = {}
    for k in episode_metrics[f"{scene_id}_{episode_id}"].keys():
        aggregated_metrics[f"{k}/mean"] = sum(
            v[k] for v in episode_metrics.values()) / len(episode_metrics)
        aggregated_metrics[f"{k}/min"] = min(
            v[k] for v in episode_metrics.values())
        aggregated_metrics[f"{k}/max"] = max(
            v[k] for v in episode_metrics.values())

    print("Per episode:")
    pprint.pprint(episode_metrics)
    print()
    print("Aggregate:")
    pprint.pprint(aggregated_metrics)


if __name__ == "__main__":
    main()
