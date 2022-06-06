from habitat.core.env import Env

from submission.utils.config_utils import get_config
from submission.agent import Agent


def main():
    config, config_str = get_config("submission/configs/config.yaml")

    agent = Agent(config=config, rank=0, ddp=False)
    env = Env(config=config.TASK_CONFIG)

    obs = env.reset()
    agent.reset()

    while not env.episode_over:
        action = agent.act(obs)
        obs = env.step(action)

    metrics = env.get_metrics()
    print(metrics)


if __name__ == "__main__":
    main()
