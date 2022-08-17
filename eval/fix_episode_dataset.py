import gzip
import json
import glob


data_dir = "habitat-challenge-data/objectgoal_hm3d/val_mini/content"
scene_dataset = "habitat-challenge-data/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

for path in glob.glob(f"{data_dir}/*"):
    with gzip.open(path, "rt") as f:
        scene_data = json.load(f)

    for episode in scene_data["episodes"]:
        episode["scene_dataset_config"] = scene_dataset

    with gzip.open(path, "w") as f:
        f.write(json.dumps(scene_data).encode())
