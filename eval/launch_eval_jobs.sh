#!/bin/bash

# Confusion matrix on HM3D, MP3D, Gibson
sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml
sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_mp3d_config.yaml
sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_gibson_config.yaml

# Ablation study on HM3D
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_standard
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_goal_on_same_floor EVAL_VECTORIZED.goal_on_same_floor 1
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_gt_semantics GROUND_TRUTH_SEMANTICS 1
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_goal_on_same_floor_and_gt_semantics EVAL_VECTORIZED.goal_on_same_floor 1 GROUND_TRUTH_SEMANTICS 1
