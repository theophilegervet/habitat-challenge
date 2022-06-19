import time
import torch
import json
from collections import defaultdict
import numpy as np
import os
import shutil
import cv2
import glob
from natsort import natsorted

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

        aggregated_metrics = defaultdict(list)
        metrics = set([k for k in list(episode_metrics.values())[0].keys()
                       if k != "goal_name"])
        for v in episode_metrics.values():
            for k in metrics:
                aggregated_metrics[f"{k}/total"].append(v[k])
                aggregated_metrics[f"{k}/{v['goal_name']}"].append(v[k])
        aggregated_metrics = dict(sorted({
            k2: v2
            for k1, v1 in aggregated_metrics.items()
            for k2, v2 in {
                f"{k1}/mean": np.mean(v1),
                f"{k1}/min": np.min(v1),
                f"{k1}/max": np.max(v1),
            }.items()
        }.items()))

        with open(f"{self.results_dir}/{split}_aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/{split}_episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)

    def record_videos(self,
                      source_dir: str,
                      target_dir: str,
                      record_planner: bool = False):

        def record_video(episode_dir: str):
            episode_name = episode_dir.split("/")[-1]
            print(f"Recording video {episode_name}")

            # Semantic map vis
            img_array = []
            filenames = natsorted(glob.glob(f"{episode_dir}/snapshot*.png"))
            if len(filenames) == 0:
                return
            for filename in filenames:
                img = cv2.imread(filename)
                height, width, _ = img.shape
                size = (width, height)
                img_array.append(img)
            out = cv2.VideoWriter(
                f"{target_dir}/{episode_name}.avi",
                cv2.VideoWriter_fourcc(*"DIVX"), 15, size
            )
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

            # Planner vis
            if record_planner:
                img_array = []
                for filename in natsorted(
                    glob.glob(f"{episode_dir}/planner_snapshot*.png")):
                    img = cv2.imread(filename)
                    height, width, _ = img.shape
                    size = (width, height)
                    img_array.append(img)
                out = cv2.VideoWriter(
                    f"{target_dir}/planner_{episode_name}.avi",
                    cv2.VideoWriter_fourcc(*"DIVX"), 15, size
                )
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()

        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        for episode_dir in glob.glob(f"{source_dir}/*"):
            record_video(episode_dir)


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/config.yaml")
    evaluator = VectorizedEvaluator(config, config_str)

    if not config.EVAL_VECTORIZED.specific_episodes:
        evaluator.eval(
            split=config.EVAL_VECTORIZED.split,
            num_episodes_per_env=config.EVAL_VECTORIZED.num_episodes_per_env
        )

    else:
        episodes = {
            "split": "val",
            "episode_keys": ['TEEsavR23oF_3', '6s7QHgap2fW_35', '6s7QHgap2fW_17', 'TEEsavR23oF_38', 'ziup5kvtCCR_62', 'TEEsavR23oF_5', 'ziup5kvtCCR_26', 'ziup5kvtCCR_97', '6s7QHgap2fW_29', 'TEEsavR23oF_84', '6s7QHgap2fW_69', 'ziup5kvtCCR_0', '6s7QHgap2fW_6', 'TEEsavR23oF_17', 'cvZr5TUy5C5_72', 'TEEsavR23oF_89', 'TEEsavR23oF_36', 'TEEsavR23oF_61', 'cvZr5TUy5C5_4', 'TEEsavR23oF_64', 'cvZr5TUy5C5_65', 'TEEsavR23oF_86', '6s7QHgap2fW_47', 'TEEsavR23oF_6', 'cvZr5TUy5C5_95', 'ziup5kvtCCR_5', 'TEEsavR23oF_74', 'TEEsavR23oF_62', 'ziup5kvtCCR_94', 'TEEsavR23oF_26', 'ziup5kvtCCR_9', 'cvZr5TUy5C5_5', 'TEEsavR23oF_73', 'ziup5kvtCCR_98', 'ziup5kvtCCR_90', 'TEEsavR23oF_13', 'cvZr5TUy5C5_16', 'ziup5kvtCCR_68', '6s7QHgap2fW_78', 'TEEsavR23oF_72', '6s7QHgap2fW_22', 'TEEsavR23oF_97', 'cvZr5TUy5C5_25', 'ziup5kvtCCR_15', 'cvZr5TUy5C5_31', 'TEEsavR23oF_35', 'TEEsavR23oF_49', 'ziup5kvtCCR_84', 'TEEsavR23oF_21', 'cvZr5TUy5C5_30', '6s7QHgap2fW_57', '6s7QHgap2fW_92']
        }
        evaluator.eval_on_specific_episodes(episodes)

    if config.EVAL_VECTORIZED.record_videos:
        evaluator.record_videos(
            source_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            target_dir="data/videos",
            record_planner=config.EVAL_VECTORIZED.record_planner_videos
        )
