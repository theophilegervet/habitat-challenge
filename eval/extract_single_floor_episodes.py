from matplotlib import pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat.core.env import Env

from submission.utils.config_utils import get_config


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/eval_hm3d_config.yaml")

    env = Env(config=config.TASK_CONFIG)
    obs = env.reset()

    # Optimal floor threshold seems to be around 1.5m
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

    total_episodes = len(env._dataset.episodes)
    env._dataset.episodes = [
        episode for episode in env._dataset.episodes
        if len([
            goal for goal in episode.goals
            if abs(episode.start_position[1] - goal.position[1]) < 1.0  # 1.5
        ]) > 0
    ]
    same_floor_episodes = len(env._dataset.episodes)
    print("Total episodes:", total_episodes)
    print("Same floor episodes:", same_floor_episodes)
    print(f"Different floor episodes {(total_episodes - same_floor_episodes) / total_episodes * 100:.2f}%")
