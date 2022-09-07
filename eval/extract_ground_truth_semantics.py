import torch
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat.core.env import Env

from submission.utils.config_utils import get_config
from submission.utils.constants import mp3d_categories_mapping


if __name__ == "__main__":
    config, config_str = get_config("submission/configs/debug_config.yaml")

    hm3d_to_mp3d_map_path = "submission/utils/matterport_category_mappings.tsv"
    df = pd.read_csv(hm3d_to_mp3d_map_path, sep='    ', header=0)

    hm3d_to_mp3d = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}

    env = Env(config=config.TASK_CONFIG)
    obs = env.reset()

    instance_id_to_category_id = torch.tensor([
        mp3d_categories_mapping.get(
            hm3d_to_mp3d.get(obj.category.name().lower().strip()),
            config.ENVIRONMENT.num_sem_categories - 1
        )
        for obj in env.sim.semantic_annotations().objects
    ])

    print(instance_id_to_category_id)
