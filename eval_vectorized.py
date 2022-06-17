import time
import torch
import os
import json
from collections import defaultdict

from habitat import Config
from habitat.core.vector_env import VectorEnv

from submission.utils.config_utils import get_config
from submission.agent import Agent
from submission.vector_env import (
    make_vector_envs,
    make_vector_envs_on_specific_episodes
)


class VectorizedEvaluator:

    def __init__(self, config: Config, config_str: str):
        self.config = config
        self.config_str = config_str

        self.results_dir = f"{config.DUMP_LOCATION}/results/{config.EXP_NAME}"
        os.makedirs(self.results_dir, exist_ok=True)

    def eval(self, split="val", num_episodes_per_env=10):
        # train split = 80 scenes with 50K episodes each (4M total)
        # val split = 20 scenes with 100 episodes each (2K total)
        assert split in ["train", "val"]
        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.SPLIT = split
        self.config.freeze()

        agent = Agent(config=self.config, rank=0, ddp=False)
        envs = make_vector_envs(self.config, max_scene_repeat_episodes=5)

        self._eval(
            agent,
            envs,
            split,
            num_episodes_per_env=num_episodes_per_env,
            episode_keys=None
        )

    def eval_on_specific_episodes(self, episodes):
        scene2episodes = defaultdict(list)
        for episode in episodes["episode_keys"]:
            scene_id, episode_id = episode.split("_")
            scene2episodes[scene_id].append(episode_id)
        scene2episodes = dict(scene2episodes)

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.SPLIT = episodes["split"]
        self.config.NUM_ENVIRONMENTS = len(scene2episodes)
        self.config.freeze()

        agent = Agent(config=self.config, rank=0, ddp=False)
        envs = make_vector_envs_on_specific_episodes(self.config, scene2episodes)

        self._eval(
            agent,
            envs,
            episodes["split"],
            num_episodes_per_env=None,
            episode_keys=set(episodes["episode_keys"])
        )

    def _eval(self,
              agent: Agent,
              envs: VectorEnv,
              split: str,
              num_episodes_per_env=None,
              episode_keys=None):

        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes) or episode_keys
        # (stop after we iterate through a list of specific episodes)
        assert ((num_episodes_per_env is not None and episode_keys is None) or
                (num_episodes_per_env is None and episode_keys is not None))

        def stop():
            if num_episodes_per_env is not None:
                return all([i >= num_episodes_per_env for i in episode_idxs])
            elif episode_keys is not None:
                return done_episode_keys == episode_keys

        start_time = time.time()
        episode_metrics = {}
        episode_idxs = [0] * envs.num_envs
        done_episode_keys = set()

        obs, infos = zip(*envs.call(["reset"] * envs.num_envs))
        agent.reset_vectorized()

        while not stop():
            # t0 = time.time()

            obs = torch.cat([ob.to(agent.device) for ob in obs])
            pose_delta = torch.cat([info["pose_delta"] for info in infos])
            goal_category = torch.cat(
                [info["goal_category"] for info in infos])

            planner_inputs, vis_inputs = agent.prepare_planner_inputs(
                obs, pose_delta, goal_category)

            # t1 = time.time()
            # print(f"[Agent] Semantic mapping and policy time: {t1 - t0:.2f}")

            obs, dones, infos = zip(*envs.call(
                ["plan_and_step"] * envs.num_envs,
                [{"planner_inputs": p_in, "vis_inputs": v_in}
                 for p_in, v_in in zip(planner_inputs, vis_inputs)]
            ))

            # t2 = time.time()
            # print(f"[Vectorized Env] Obs preprocessing, planning, "
            #       f"and step time: {t2 - t1:.2f}")
            # print(f"Total time: {t2 - t0:.2f}")
            # print()

            # For done episodes, gather statistics and reset agent —
            # the environment itself is automatically reset by its
            # wrapper
            for e, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    episode_key = (f"{info['last_episode_scene_id']}_"
                                   f"{info['last_episode_id']}")

                    # If the episode keys we care about are specified,
                    #  ignore all other episodes
                    if episode_keys is not None:
                        if episode_key in episode_keys:
                            done_episode_keys.add(episode_key)
                            episode_metrics[episode_key] = {
                                **info["last_episode_metrics"],
                                "goal_name": info["last_goal_name"]
                            }
                            print(
                                f"Finished episode {episode_key} after "
                                f"{round(time.time() - start_time, 2)} seconds")

                    elif num_episodes_per_env is not None:
                        if episode_idxs[e] < num_episodes_per_env:
                            episode_metrics[episode_key] = {
                                **info["last_episode_metrics"],
                                "goal_name": info["last_goal_name"]
                            }
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs} / {num_episodes_per_env} "
                            f"after {round(time.time() - start_time, 2)} seconds")

                    agent.reset_vectorized_for_env(e)

        envs.close()

        aggregated_metrics = {}
        for k in list(episode_metrics.values())[0].keys():
            if k == "goal_name":
                continue
            aggregated_metrics[f"{k}/mean"] = sum(
                v[k] for v in episode_metrics.values()) / len(episode_metrics)
            aggregated_metrics[f"{k}/min"] = min(
                v[k] for v in episode_metrics.values())
            aggregated_metrics[f"{k}/max"] = max(
                v[k] for v in episode_metrics.values())

        with open(f"{self.results_dir}/{split}_aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/{split}_episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/config.yaml")
    evaluator = VectorizedEvaluator(config, config_str)

    # evaluator.eval(split="train", num_episodes_per_env=5 * 10)
    evaluator.eval(split="val", num_episodes_per_env=1)

    # episodes = {
    #     "split": "val",
    #     "episode_keys": [
    #         # too far
    #         "6s7QHgap2fW_6",  # success 12
    #         "mL8ThkuaVTM_7",  # x => this episode seems buggy
    #         "mv2HUxq3B53_45", # x => hard, the tv is positioned weirdly
    #         "svBbv1Pavdk_53", # x 12, planner gets stuck
    #         "svBbv1Pavdk_66", # success 12
    #         "zt1RVoi7PcG_27", # x 12, segmentation fp
    #         "zt1RVoi7PcG_48", # success 12
    #         # success
    #         "4ok3usBNeis_42", # success 12
    #         "6s7QHgap2fW_86", # success 12
    #         "mL8ThkuaVTM_4",  # x 12, frontier exploration messes up
    #     ]
    # }
    # evaluator.eval_on_specific_episodes(episodes)
