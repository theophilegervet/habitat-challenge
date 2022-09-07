import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat.core.env import Env

from submission.utils.config_utils import get_config


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/debug_config.yaml")
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.freeze()

    env = Env(config=config.TASK_CONFIG)
    obs = env.reset()
    print(env.sim.semantic_annotations().objects)

    # scene_id = env.current_episode.scene_id.split("/")[-1].split(".")[0]
    # episode_id = env.current_episode.episode_id

    # Set mapping to convert instance segmentation to semantic segmentation
    # when using ground-truth semantics
    # instance_id_to_category_id = torch.tensor([
    #     hm3d_categories_mapping.get(
    #         obj.category.index(),
    #         config.ENVIRONMENT.num_sem_categories - 1
    #     )
    #     for obj in env.sim.semantic_annotations().objects
    # ])
    #
    # instance_id_to_category_id = torch.tensor([
    #     obj.category.index()
    #     for obj in self.habitat_env.sim.semantic_annotations().objects
    # ], device=self.device)
