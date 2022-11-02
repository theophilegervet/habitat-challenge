from matplotlib import pyplot as plt
from pathlib import Path
import os
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat.core.env import Env

from submission.utils.config_utils import get_config


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    config, config_str = get_config("submission/configs/eval_hm3d_config.yaml")

    env = Env(config=config.TASK_CONFIG)
    obs = env.reset()

    episode_start_height_and_goal_heights = [
        (
            episode.start_position[1],
            [goal.position[1] for goal in episode.goals]
        )
        for episode in env._dataset.episodes
    ]
    episode_start_to_goal_height_distances = [
        abs(start_height - goal_height)
        for start_height, goal_heights in episode_start_height_and_goal_heights
        for goal_height in goal_heights
    ]
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.hist(episode_start_to_goal_height_distances, bins=100)
    plt.show()

    num_total_episodes = len(env._dataset.episodes)
    env._dataset.episodes = [
        episode for episode in env._dataset.episodes
        if len([
            goal for goal in episode.goals
            if episode.start_position[1] - 0.25 < goal.position[1] < episode.start_position[1] + 1.5
        ]) > 0
    ]
    num_same_floor_episodes = len(env._dataset.episodes)

    first_floor_episodes = []
    for episode in env._dataset.episodes:
        scene_dir = "/".join(episode.scene_id.split("/")[:-1])
        map_dir = scene_dir + "/floor_semantic_maps_annotations_top_down"
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]
        with open(f"{map_dir}/{scene_id}_info.json", "r") as f:
            scene_info = json.load(f)
        print("episode.start_position[1] * 100.", episode.start_position[1] * 100.)
        print('scene_info["floor_heights_cm"][0]', scene_info["floor_heights_cm"][0])
        print()
        if abs(episode.start_position[1] * 100. - scene_info["floor_heights_cm"][0]) < 0.5:
            first_floor_episodes.append(episode)

    print("Total episodes:", num_total_episodes)
    print("Same floor episodes:", num_same_floor_episodes)
    print("First floor episodes:", len(first_floor_episodes))
    print(f"Different floor episodes {(num_total_episodes - num_same_floor_episodes) / num_total_episodes * 100:.2f}%")
    print(f"Not first floor episodes {(num_total_episodes - len(first_floor_episodes)) / num_total_episodes * 100:.2f}%")
