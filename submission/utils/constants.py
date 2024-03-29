from pathlib import Path
import pandas as pd


MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001

challenge_goal_name_to_goal_name = {
    "chair": "chair",
    "sofa": "couch",
    "plant": "potted plant",
    "bed": "bed",
    "toilet": "toilet",
    "tv_monitor": "tv"
}

goal_id_to_goal_name = {
    0: "chair",
    1: "bed",
    2: "potted plant",
    3: "toilet",
    4: "tv",
    5: "couch",
}

goal_id_to_coco_id = {
    0: 0,  # chair
    1: 3,  # bed
    2: 2,  # potted plant
    3: 4,  # toilet
    4: 5,  # tv
    5: 1,  # couch
}

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    "no-category": 15,
}

# detectron2 model trained on COCO
detectron2_categories_mapping = {
    56: 0,   # chair
    57: 1,   # couch
    58: 2,   # plant
    59: 3,   # bed
    61: 4,   # toilet
    62: 5,   # tv
    60: 6,   # table
    69: 7,   # oven
    71: 8,   # sink
    72: 9,   # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

# mmdetection model trained on HM3D
mmdetection_categories_mapping = {
    0: 0,   # chair
    5: 1,   # couch
    8: 2,   # plant
    6: 3,   # bed
    10: 4,  # toilet
    13: 5,  # tv
    1: 6,   # table
    9: 8,   # sink
}

# ground-truth HM3D semantics are mapped to MP3D categories
mp3d_categories_mapping = {
    3: 0,   # chair
    10: 1,  # couch
    14: 2,  # plant
    11: 3,  # bed
    18: 4,  # toilet
    22: 5,  # tv
    5: 6,   # table
    15: 8,  # sink

}

hm3d_to_mp3d_path = Path(__file__).resolve().parent / "matterport_category_mappings.tsv"
df = pd.read_csv(hm3d_to_mp3d_path, sep='    ', header=0)
hm3d_to_mp3d = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}

coco_categories_color_palette = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # couch
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # potted plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # dining-table
    0.66,
    0.9232,
    0.9400000000000001,  # oven
    0.66,
    0.8182,
    0.9400000000000001,  # sink
    0.66,
    0.7132,
    0.9400000000000001,  # refrigerator
    0.7117999999999999,
    0.66,
    0.9400000000000001,  # book
    0.8168,
    0.66,
    0.9400000000000001,  # clock
    0.9218,
    0.66,
    0.9400000000000001,  # vase
    0.9400000000000001,
    0.66,
    0.8531999999999998,  # cup
    0.9400000000000001,
    0.66,
    0.748199999999999,  # bottle
]

# TODO Change goal color
map_color_palette = [
    1.0,
    1.0,
    1.0,  # empty space
    0.6,
    0.6,
    0.6,  # obstacles
    0.95,
    0.95,
    0.95,  # explored area
    0.96,
    0.36,
    0.26,  # visited area & goal
    *coco_categories_color_palette,
]

frame_color_palette = [
    *coco_categories_color_palette,
    1.0,
    1.0,
    1.0,  # no category
]
