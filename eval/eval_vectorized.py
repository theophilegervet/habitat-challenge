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
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat import Config
from habitat.core.vector_env import VectorEnv

from submission.utils.config_utils import get_config
from submission.agent import Agent
from submission.env_wrapper import (
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="submission/configs/debug_config.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    print("Config:")
    config, config_str = get_config(args.config_path, args.opts)
    evaluator = VectorizedEvaluator(config, config_str)
    print(config_str)
    print("-" * 100)

    if not config.EVAL_VECTORIZED.specific_episodes:
        evaluator.eval(
            split=config.EVAL_VECTORIZED.split,
            num_episodes_per_env=config.EVAL_VECTORIZED.num_episodes_per_env
        )

    else:
        episodes = {
            "split": "val",
            # "episode_keys": [
            #     "6s7QHgap2fW_6",
            #     "6s7QHgap2fW_29",
            #     "mv2HUxq3B53_3",
            #     "mv2HUxq3B53_60",
            # ]
            # "episode_keys": ['ziup5kvtCCR_59', 'mL8ThkuaVTM_51', 'ziup5kvtCCR_81', 'mL8ThkuaVTM_34', '5cdEh9F2hJL_64', 'mL8ThkuaVTM_26', 'zt1RVoi7PcG_60', 'p53SfW6mjZe_86', 'wcojb4TFT35_4', 'XB4GS9ShBRE_15', 'Nfvxx8J5NCo_2', '6s7QHgap2fW_9', 'ziup5kvtCCR_89', 'zt1RVoi7PcG_49', 'ziup5kvtCCR_14', 'DYehNKdT76V_17', 'bxsVRursffK_53', '5cdEh9F2hJL_82', '4ok3usBNeis_11', 'q3zU7Yy5E5s_70', 'TEEsavR23oF_21', 'qyAac8rV8Zk_31', 'Nfvxx8J5NCo_7', 'Nfvxx8J5NCo_20', 'p53SfW6mjZe_40', 'wcojb4TFT35_78', '6s7QHgap2fW_3', 'q3zU7Yy5E5s_0', 'ziup5kvtCCR_47', 'QaLdnwvtxbs_60', 'XB4GS9ShBRE_6', 'zt1RVoi7PcG_44', 'QaLdnwvtxbs_78', 'Nfvxx8J5NCo_5', 'DYehNKdT76V_87', 'TEEsavR23oF_65', 'mv2HUxq3B53_74', 'p53SfW6mjZe_3', 'wcojb4TFT35_48', 'qyAac8rV8Zk_43', '5cdEh9F2hJL_1', '6s7QHgap2fW_34', 'bxsVRursffK_49', 'mL8ThkuaVTM_64', 'Nfvxx8J5NCo_77', 'q3zU7Yy5E5s_84', 'DYehNKdT76V_86', 'svBbv1Pavdk_83', 'p53SfW6mjZe_66', '5cdEh9F2hJL_73', '5cdEh9F2hJL_14', '6s7QHgap2fW_55', 'bxsVRursffK_37', 'ziup5kvtCCR_56', 'Nfvxx8J5NCo_98', 'DYehNKdT76V_55', 'q3zU7Yy5E5s_77', 'mL8ThkuaVTM_52', 'mL8ThkuaVTM_22', 'svBbv1Pavdk_19', '5cdEh9F2hJL_17', 'QaLdnwvtxbs_1', 'qyAac8rV8Zk_66', '6s7QHgap2fW_1', '4ok3usBNeis_96', 'zt1RVoi7PcG_4', 'bxsVRursffK_67', 'mL8ThkuaVTM_79', 'wcojb4TFT35_42', 'q3zU7Yy5E5s_83', 'q3zU7Yy5E5s_7', '5cdEh9F2hJL_0', 'qyAac8rV8Zk_80', '6s7QHgap2fW_26', 'ziup5kvtCCR_78', 'p53SfW6mjZe_88', 'q3zU7Yy5E5s_34', 'mL8ThkuaVTM_65', 'mv2HUxq3B53_85', 'Nfvxx8J5NCo_87', 'ziup5kvtCCR_61', 'TEEsavR23oF_85', 'ziup5kvtCCR_49', 'QaLdnwvtxbs_34', 'Nfvxx8J5NCo_68', 'zt1RVoi7PcG_85', 'wcojb4TFT35_47', '6s7QHgap2fW_0', 'p53SfW6mjZe_73', 'svBbv1Pavdk_79', 'qyAac8rV8Zk_24', 'TEEsavR23oF_68', 'q3zU7Yy5E5s_27', 'mL8ThkuaVTM_7', 'Nfvxx8J5NCo_17', 'wcojb4TFT35_7', 'qyAac8rV8Zk_2', 'Dd4bFSTQ8gi_71', 'mL8ThkuaVTM_41', 'QaLdnwvtxbs_35', 'svBbv1Pavdk_4', 'XB4GS9ShBRE_30', 'Nfvxx8J5NCo_66', '6s7QHgap2fW_7', 'zt1RVoi7PcG_10', 'ziup5kvtCCR_74', 'QaLdnwvtxbs_44', 'bxsVRursffK_39', 'qyAac8rV8Zk_59', '5cdEh9F2hJL_24', 'TEEsavR23oF_89', 'mL8ThkuaVTM_24', 'Nfvxx8J5NCo_79', 'XB4GS9ShBRE_35', 'svBbv1Pavdk_69', 'p53SfW6mjZe_15', 'DYehNKdT76V_24', '6s7QHgap2fW_93', 'zt1RVoi7PcG_113', 'bxsVRursffK_35', 'svBbv1Pavdk_41', '5cdEh9F2hJL_21', 'wcojb4TFT35_2', 'QaLdnwvtxbs_85', 'wcojb4TFT35_22', '5cdEh9F2hJL_90', 'ziup5kvtCCR_50', 'q3zU7Yy5E5s_98', 'XB4GS9ShBRE_80', 'mL8ThkuaVTM_35', 'qyAac8rV8Zk_7', 'Nfvxx8J5NCo_85', 'Nfvxx8J5NCo_83', 'mv2HUxq3B53_14', 'p53SfW6mjZe_87', 'Nfvxx8J5NCo_64', 'zt1RVoi7PcG_71', 'cvZr5TUy5C5_52', 'ziup5kvtCCR_60', '6s7QHgap2fW_17', '5cdEh9F2hJL_72', 'QaLdnwvtxbs_71', 'XB4GS9ShBRE_16', 'qyAac8rV8Zk_96', 'ziup5kvtCCR_32', 'q3zU7Yy5E5s_48', 'DYehNKdT76V_65', 'p53SfW6mjZe_57', 'zt1RVoi7PcG_46', 'ziup5kvtCCR_88', '5cdEh9F2hJL_86', '4ok3usBNeis_50', 'wcojb4TFT35_37', '6s7QHgap2fW_96', 'XB4GS9ShBRE_72', '5cdEh9F2hJL_8', 'DYehNKdT76V_28', 'Nfvxx8J5NCo_9', 'bxsVRursffK_52', 'p53SfW6mjZe_12', 'q3zU7Yy5E5s_36', 'qyAac8rV8Zk_53', 'Nfvxx8J5NCo_6', 'QaLdnwvtxbs_32', 'zt1RVoi7PcG_69', '6s7QHgap2fW_87', 'DYehNKdT76V_81', '4ok3usBNeis_72', 'Dd4bFSTQ8gi_82', 'p53SfW6mjZe_48', 'XB4GS9ShBRE_36', 'qyAac8rV8Zk_58', 'wcojb4TFT35_93', 'qyAac8rV8Zk_67', 'QaLdnwvtxbs_61', 'bxsVRursffK_83', 'TEEsavR23oF_53', 'Nfvxx8J5NCo_74', '6s7QHgap2fW_59', '6s7QHgap2fW_13', 'TEEsavR23oF_54', '5cdEh9F2hJL_51', 'ziup5kvtCCR_48', 'p53SfW6mjZe_70', 'zt1RVoi7PcG_55', 'XB4GS9ShBRE_70', 'bxsVRursffK_13', '5cdEh9F2hJL_68', 'ziup5kvtCCR_73', 'q3zU7Yy5E5s_5', 'bxsVRursffK_21', 'wcojb4TFT35_44', 'DYehNKdT76V_94', 'p53SfW6mjZe_67', 'svBbv1Pavdk_84', 'qyAac8rV8Zk_21', 'ziup5kvtCCR_3', 'ziup5kvtCCR_42', '6s7QHgap2fW_25', 'ziup5kvtCCR_52', 'Nfvxx8J5NCo_92', '4ok3usBNeis_21', 'TEEsavR23oF_77', 'wcojb4TFT35_33', 'DYehNKdT76V_30', 'p53SfW6mjZe_5', 'QaLdnwvtxbs_10', 'svBbv1Pavdk_21', 'bxsVRursffK_42', 'zt1RVoi7PcG_78', 'XB4GS9ShBRE_56', '5cdEh9F2hJL_75', 'Nfvxx8J5NCo_15', '6s7QHgap2fW_88', 'ziup5kvtCCR_41', '4ok3usBNeis_29', 'zt1RVoi7PcG_3', 'TEEsavR23oF_6', '4ok3usBNeis_24', 'svBbv1Pavdk_75', 'bxsVRursffK_2', '5cdEh9F2hJL_45', 'mv2HUxq3B53_96', 'DYehNKdT76V_26', 'QaLdnwvtxbs_15', 'ziup5kvtCCR_91', 'q3zU7Yy5E5s_59', 'p53SfW6mjZe_16', 'Nfvxx8J5NCo_43', 'zt1RVoi7PcG_91', 'svBbv1Pavdk_45', 'TEEsavR23oF_11', 'XB4GS9ShBRE_64', 'wcojb4TFT35_57', 'QaLdnwvtxbs_9', '5cdEh9F2hJL_70', '4ok3usBNeis_49', 'ziup5kvtCCR_13', '6s7QHgap2fW_32', 'svBbv1Pavdk_3', 'DYehNKdT76V_60', '5cdEh9F2hJL_30', 'QaLdnwvtxbs_56', 'ziup5kvtCCR_96', 'cvZr5TUy5C5_54', 'p53SfW6mjZe_22', 'zt1RVoi7PcG_41', 'svBbv1Pavdk_67', 'wcojb4TFT35_68', '6s7QHgap2fW_24', 'QaLdnwvtxbs_29', 'DYehNKdT76V_47', '5cdEh9F2hJL_44', 'XB4GS9ShBRE_29', 'p53SfW6mjZe_34', 'ziup5kvtCCR_93', '5cdEh9F2hJL_25', 'Nfvxx8J5NCo_8', 'svBbv1Pavdk_8', 'wcojb4TFT35_70', 'q3zU7Yy5E5s_67', '6s7QHgap2fW_94', 'Nfvxx8J5NCo_57', 'Nfvxx8J5NCo_72', 'zt1RVoi7PcG_39', 'DYehNKdT76V_43', 'QaLdnwvtxbs_41', 'svBbv1Pavdk_74', 'wcojb4TFT35_25', '6s7QHgap2fW_82', 'p53SfW6mjZe_53', 'Nfvxx8J5NCo_30', '5cdEh9F2hJL_29', 'QaLdnwvtxbs_67', 'Nfvxx8J5NCo_4', '5cdEh9F2hJL_37', '6s7QHgap2fW_39', 'TEEsavR23oF_7', 'wcojb4TFT35_89', 'zt1RVoi7PcG_101', 'ziup5kvtCCR_44', 'svBbv1Pavdk_94', 'p53SfW6mjZe_50', 'XB4GS9ShBRE_74', 'q3zU7Yy5E5s_12', 'wcojb4TFT35_38', '6s7QHgap2fW_4', '5cdEh9F2hJL_13', 'ziup5kvtCCR_2', 'wcojb4TFT35_0', 'QaLdnwvtxbs_57', 'p53SfW6mjZe_25', 'svBbv1Pavdk_1', 'TEEsavR23oF_61', 'XB4GS9ShBRE_44', 'mv2HUxq3B53_4', 'q3zU7Yy5E5s_72', 'Nfvxx8J5NCo_29', 'zt1RVoi7PcG_111', 'mv2HUxq3B53_39', 'wcojb4TFT35_66', '5cdEh9F2hJL_3', 'Nfvxx8J5NCo_93', 'p53SfW6mjZe_21', 'QaLdnwvtxbs_88', 'svBbv1Pavdk_12', 'DYehNKdT76V_34', 'wcojb4TFT35_17', '5cdEh9F2hJL_28', '5cdEh9F2hJL_16', '5cdEh9F2hJL_10', 'ziup5kvtCCR_83', 'ziup5kvtCCR_51', 'Nfvxx8J5NCo_21', 'svBbv1Pavdk_86', 'Nfvxx8J5NCo_78', 'wcojb4TFT35_12', '6s7QHgap2fW_56', 'DYehNKdT76V_98', '5cdEh9F2hJL_43', 'p53SfW6mjZe_7', 'mv2HUxq3B53_86', 'svBbv1Pavdk_36', 'zt1RVoi7PcG_21', 'ziup5kvtCCR_43', '5cdEh9F2hJL_59', '6s7QHgap2fW_66', 'Nfvxx8J5NCo_40', 'zt1RVoi7PcG_31', 'wcojb4TFT35_15', 'p53SfW6mjZe_94', 'ziup5kvtCCR_22', '6s7QHgap2fW_69', 'svBbv1Pavdk_9', 'QaLdnwvtxbs_66', '5cdEh9F2hJL_77', 'DYehNKdT76V_10', 'wcojb4TFT35_96', 'p53SfW6mjZe_55', 'wcojb4TFT35_59', '6s7QHgap2fW_67', 'svBbv1Pavdk_25', '5cdEh9F2hJL_89', '5cdEh9F2hJL_58', 'QaLdnwvtxbs_72', 'zt1RVoi7PcG_108', '5cdEh9F2hJL_91', 'wcojb4TFT35_28', 'mv2HUxq3B53_35', '6s7QHgap2fW_5', 'svBbv1Pavdk_38', 'ziup5kvtCCR_40', 'DYehNKdT76V_18', 'QaLdnwvtxbs_86', 'wcojb4TFT35_11', '6s7QHgap2fW_72', 'p53SfW6mjZe_68', 'DYehNKdT76V_46', 'wcojb4TFT35_30', 'QaLdnwvtxbs_65', 'ziup5kvtCCR_39', 'zt1RVoi7PcG_81', '6s7QHgap2fW_37', 'svBbv1Pavdk_20', 'p53SfW6mjZe_11', 'ziup5kvtCCR_21', 'wcojb4TFT35_87', 'ziup5kvtCCR_54', '6s7QHgap2fW_77', 'p53SfW6mjZe_20', 'wcojb4TFT35_16', '6s7QHgap2fW_70', 'zt1RVoi7PcG_68', 'ziup5kvtCCR_30', 'wcojb4TFT35_73', 'p53SfW6mjZe_9', 'svBbv1Pavdk_24', 'ziup5kvtCCR_66', 'zt1RVoi7PcG_14', 'QaLdnwvtxbs_89', 'wcojb4TFT35_58', 'QaLdnwvtxbs_23', 'DYehNKdT76V_54', 'ziup5kvtCCR_63', 'p53SfW6mjZe_36', 'DYehNKdT76V_35', 'wcojb4TFT35_85', 'svBbv1Pavdk_96', 'p53SfW6mjZe_59', 'ziup5kvtCCR_70', 'wcojb4TFT35_92', 'ziup5kvtCCR_58', 'svBbv1Pavdk_68', 'p53SfW6mjZe_24', 'wcojb4TFT35_54', 'svBbv1Pavdk_55', 'p53SfW6mjZe_74', 'wcojb4TFT35_67', 'DYehNKdT76V_76', 'wcojb4TFT35_19', 'DYehNKdT76V_95', 'svBbv1Pavdk_18', 'wcojb4TFT35_14', 'wcojb4TFT35_90', 'wcojb4TFT35_88', 'wcojb4TFT35_69', 'wcojb4TFT35_81', 'wcojb4TFT35_52', 'wcojb4TFT35_31', 'wcojb4TFT35_76', 'wcojb4TFT35_49']
            "episode_keys": ['ziup5kvtCCR_61', 'Nfvxx8J5NCo_87', '6s7QHgap2fW_25', 'mv2HUxq3B53_74', 'mv2HUxq3B53_35', 'Nfvxx8J5NCo_4', 'TEEsavR23oF_7', 'zt1RVoi7PcG_108', 'p53SfW6mjZe_36', 'p53SfW6mjZe_70', 'zt1RVoi7PcG_55', 'p53SfW6mjZe_57', 'Nfvxx8J5NCo_68', 'TEEsavR23oF_6', 'XB4GS9ShBRE_64', 'wcojb4TFT35_59', 'DYehNKdT76V_28', 'q3zU7Yy5E5s_77', 'Nfvxx8J5NCo_43', 'wcojb4TFT35_67', 'wcojb4TFT35_17', '5cdEh9F2hJL_16', 'DYehNKdT76V_87', 'svBbv1Pavdk_8', '5cdEh9F2hJL_45', 'q3zU7Yy5E5s_72', 'svBbv1Pavdk_68', 'DYehNKdT76V_17', 'wcojb4TFT35_33', '6s7QHgap2fW_94', 'bxsVRursffK_21', '6s7QHgap2fW_4', 'mL8ThkuaVTM_26', '6s7QHgap2fW_24', 'wcojb4TFT35_49', 'ziup5kvtCCR_51', 'svBbv1Pavdk_12', 'QaLdnwvtxbs_23', 'DYehNKdT76V_81', '6s7QHgap2fW_7', 'q3zU7Yy5E5s_5', 'QaLdnwvtxbs_89', 'cvZr5TUy5C5_54', 'p53SfW6mjZe_67', 'qyAac8rV8Zk_67', 'wcojb4TFT35_70', 'zt1RVoi7PcG_39', '5cdEh9F2hJL_0', 'qyAac8rV8Zk_43', 'q3zU7Yy5E5s_59', '6s7QHgap2fW_87', 'mL8ThkuaVTM_64', 'wcojb4TFT35_15', 'q3zU7Yy5E5s_36', 'ziup5kvtCCR_52', '6s7QHgap2fW_66', 'ziup5kvtCCR_63', 'wcojb4TFT35_28', 'Nfvxx8J5NCo_6', 'XB4GS9ShBRE_74', '5cdEh9F2hJL_58', 'zt1RVoi7PcG_31', 'qyAac8rV8Zk_53', 'DYehNKdT76V_94', '5cdEh9F2hJL_75', 'wcojb4TFT35_11', 'XB4GS9ShBRE_72', 'TEEsavR23oF_77', 'QaLdnwvtxbs_60', 'zt1RVoi7PcG_14', 'mL8ThkuaVTM_52', 'DYehNKdT76V_24', 'Nfvxx8J5NCo_98', 'p53SfW6mjZe_88', '5cdEh9F2hJL_24', 'wcojb4TFT35_2', 'p53SfW6mjZe_3', 'ziup5kvtCCR_42', 'q3zU7Yy5E5s_84', 'p53SfW6mjZe_21', 'ziup5kvtCCR_60', 'q3zU7Yy5E5s_48', 'wcojb4TFT35_47', 'DYehNKdT76V_26', 'Nfvxx8J5NCo_40', 'zt1RVoi7PcG_3', 'QaLdnwvtxbs_78', 'ziup5kvtCCR_56', 'QaLdnwvtxbs_57', 'mL8ThkuaVTM_7', '6s7QHgap2fW_32', '5cdEh9F2hJL_90', 'XB4GS9ShBRE_56', 'svBbv1Pavdk_19', 'DYehNKdT76V_18', 'bxsVRursffK_67', 'wcojb4TFT35_69', 'zt1RVoi7PcG_46', 'svBbv1Pavdk_1', 'p53SfW6mjZe_55']
            # "episode_keys": ['QaLdnwvtxbs_10']
        }
        evaluator.eval_on_specific_episodes(episodes)

    if config.EVAL_VECTORIZED.record_videos:
        evaluator.record_videos(
            source_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            target_dir="data/videos",
            record_planner=config.EVAL_VECTORIZED.record_planner_videos
        )
