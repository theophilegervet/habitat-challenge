import time
from pathlib import Path
import pprint

from habitat import Config

from submission.utils.config_utils import get_config
from submission.agent import Agent
from evaluation.utils import make_vector_envs


class VectorizedEvaluator:

    def __init__(self, config: Config, config_str: str):
        self.config = config
        self.config_str = config_str
        self.agent = Agent(config=config, rank=0, ddp=False)

    def eval(self, num_episodes=30):
        start_time = time.time()
        episode_metrics = {}
        episode_idx = 0

        envs = make_vector_envs(self.config)

        obs = envs.reset()
        infos = [{"last_action": None}] * len(obs)
        self.agent.reset_vectorized()

        while episode_idx < num_episodes:
            t0 = time.time()
            planner_inputs, vis_inputs = self.agent.act_vectorized(obs, infos)

            t1 = time.time()
            obs, dones, infos = zip(*envs.call(
                ["plan_and_step"] * envs.num_envs,
                [{"planner_inputs": p_in, "vis_inputs": v_in}
                 for p_in, v_in in zip(planner_inputs, vis_inputs)]
            ))
            t2 = time.time()
            print(f"[Vectorized Env] Planning and step time: {t2 - t1:.2f}")
            print(f"Total time: {t2 - t0:.2f}")
            print()

            # For done episodes, gather statistics and reset agent —
            # the environment itself is automatically reset by the
            # vectorized environment
            for e, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    episode_key = (f"{info['last_episode_scene_id']}_"
                                   f"{info['last_episode_id']}")
                    episode_metrics[episode_key] = info["last_episode_metrics"]
                    self.agent.reset_vectorized_for_env(e)
                    episode_idx += 1
                    print(f"Finished episode {episode_idx} after "
                          f"{round(time.time() - start_time, 2)} seconds")

        envs.close()

        aggregated_metrics = {}
        for k in episode_metrics[episode_key].keys():
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
    config_path = str(Path().absolute() / "submission/configs/config.yaml")
    config, config_str = get_config(config_path)
    evaluator = VectorizedEvaluator(config, config_str)
    evaluator.eval()
