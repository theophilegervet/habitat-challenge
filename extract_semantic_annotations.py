from habitat.core.env import Env

from submission.utils.config_utils import get_config


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/debug_config.yaml")
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.freeze()

    env = Env(config=config.TASK_CONFIG)
