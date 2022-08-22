from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from habitat import make_dataset

from submission.utils.config_utils import get_config


config, config_str = get_config("submission/configs/ppo_debug_config.yaml")

dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE,
                       config=config.TASK_CONFIG.DATASET)
print(dataset)
