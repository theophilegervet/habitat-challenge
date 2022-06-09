from typing import List
import random

from habitat import Config
from habitat import make_dataset
from habitat.core.dataset import ALL_SCENES_MASK
from habitat.core.env import Env
from habitat.core.vector_env import VectorEnv

from submission.utils.config_utils import get_config
from submission.agent import Agent


def _get_env_gpus(config: Config, rank: int) -> List[int]:
    """Get GPUs assigned to environments of a particular agent process."""
    num_agent_processes = len(config.AGENT_GPU_IDS)
    num_env_gpus = len(config.SIMULATOR_GPU_IDS)
    num_env_gpus_per_agent_process = num_env_gpus // num_agent_processes
    assert num_agent_processes > 0
    assert (num_env_gpus >= num_agent_processes and
            num_env_gpus % num_agent_processes == 0)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    gpus = chunks(config.SIMULATOR_GPU_IDS, num_env_gpus_per_agent_process)[rank]
    return gpus


def make_vector_envs(
        config: Config,
        auto_reset_done: bool = True,
        max_scene_repeat_episodes: int = -1
    ) -> VectorEnv:
    """Create vectorized online acting environments and split scenes
    across environments.

    Arguments:
        auto_reset_done: if True, automatically reset the environment when
         an episode is over
        max_scene_repeat_episodes: if > 0, this is the maximum number of
         consecutive episodes in the same scene — set to 1 to get some
         scene diversity in visualization but keep to -1 default for
         training as switching scenes adds overhead to the simulator
    """
    gpus = _get_env_gpus(config, rank=0)
    num_gpus = len(gpus)
    num_envs = config.NUM_ENVIRONMENTS
    assert (num_envs >= num_gpus and num_envs % num_gpus == 0)
    num_envs_per_gpu = num_envs // num_gpus

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError("No scenes to load")
        elif len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError("Not enough scenes for envs")
        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
        assert sum(map(len, scene_splits)) == len(scenes)

    configs = []
    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j
            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]
            if proc_config.NO_GPU:
                task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]
            task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
            task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = max_scene_repeat_episodes
            proc_config.freeze()
            configs.append(proc_config)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        auto_reset_done=auto_reset_done,
        env_fn_args=tuple([(configs[rank],)
                           for rank in range(len(configs))])
    )
    return envs


def make_env_fn(config):
    return Env(config.TASK_CONFIG)


class VectorizedEvaluator:

    def __init__(self, config: Config, config_str: str):
        self.config = config
        self.config_str = config_str
        self.agent = Agent(config=config, rank=0, ddp=False)

    def eval(self):
        envs = make_vector_envs(self.config, auto_reset_done=True)

        # TODO Write vectorized evaluation loop here

        envs.close()


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/config.yaml")
    evaluator = VectorizedEvaluator(config, config_str)
    evaluator.eval()
