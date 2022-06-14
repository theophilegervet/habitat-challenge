import torch
import pprint
import time

from habitat.core.env import Env
from habitat import Config

from submission.utils.config_utils import get_config
from submission.utils.constants import hm3d_categories_mapping
from submission.agent import Agent


class UnvectorizedEvaluator:

    def __init__(self, config: Config, config_str: str):
        self.config = config
        self.config_str = config_str
        self.agent = Agent(config=config, rank=0, ddp=False)

    def eval(self):
        start_time = time.time()
        episode_metrics = {}

        env = Env(config=config.TASK_CONFIG)

        from collections import defaultdict
        episodes = defaultdict(list)
        for ep in env.episodes:
            episodes[ep.scene_id].append(ep.episode_id)
        print("config.TASK_CONFIG.DATASET.SPLIT", config.TASK_CONFIG.DATASET.SPLIT)
        print("Scenes", len(episodes))
        print("Episodes per scene", [len(v) for v in episodes.values()])

        return

        for episode_idx in range(len(env.episodes)):
            obs = env.reset()
            self.agent.reset()

            scene_id = env.current_episode.scene_id.split("/")[-1].split(".")[0]
            episode_id = env.current_episode.episode_id
            self.agent.set_vis_dir(scene_id=scene_id, episode_id=episode_id)

            # Set mapping to convert instance segmentation to semantic segmentation
            # when using ground-truth semantics
            # self.agent.obs_preprocessor.set_instance_id_to_category_id(torch.tensor([
            #     hm3d_categories_mapping.get(
            #         obj.category.index(),
            #         config.ENVIRONMENT.num_sem_categories - 1
            #     )
            #     for obj in env.sim.semantic_annotations().objects
            # ]))

            while not env.episode_over:
                action = self.agent.act(obs)
                obs = env.step(action)

            print(f"Finished episode {episode_idx} after "
                  f"{round(time.time() - start_time, 2)} seconds")
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
    config, config_str = get_config("submission/configs/config.yaml")
    assert config.NUM_ENVIRONMENTS == 1
    evaluator = UnvectorizedEvaluator(config, config_str)
    evaluator.eval()
