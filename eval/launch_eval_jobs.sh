#!/bin/bash

# Confusion matrix on HM3D, MP3D, Gibson
DATASETS=("hm3d" "mp3d" "gibson")
#CHECKPOINTS=(
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_bcfe1_00000_0_2022-09-30_03-00-40/checkpoint_000800/checkpoint-800"
#  "/private/home/theop123/ray_results/ppo_train_mp3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_bcae0_00000_0_2022-10-01_08-00-07/checkpoint_000300/checkpoint-300"
#  "/private/home/theop123/ray_results/ppo_train_gibson_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_040bb_00000_0_2022-09-30_05-18-39/checkpoint_000900/checkpoint-900"
#)
CHECKPOINTS=(
  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_d3d73_00000_0_2022-10-03_11-40-18/checkpoint_001000/checkpoint-1000"
  "/private/home/theop123/ray_results/ppo_train_mp3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_bcae0_00000_0_2022-10-01_08-00-07/checkpoint_000300/checkpoint-300"
  "/private/home/theop123/ray_results/ppo_train_gibson_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_040bb_00000_0_2022-09-30_05-18-39/checkpoint_000900/checkpoint-900"
)
for i in ${!DATASETS[@]}; do
  for j in ${!DATASETS[@]}; do
    sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_${DATASETS[$i]}_config.yaml \
      AGENT.POLICY.type semantic \
      AGENT.POLICY.SEMANTIC.checkpoint_path ${CHECKPOINTS[$j]} \
      EXP_NAME eval_${DATASETS[$j]}_on_${DATASETS[$i]}
  done
done

# Ablation study on HM3D
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_standard
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_goal_on_same_floor EVAL_VECTORIZED.goal_on_same_floor 1
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_gt_semantics GROUND_TRUTH_SEMANTICS 1
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME eval_goal_on_same_floor_and_gt_semantics EVAL_VECTORIZED.goal_on_same_floor 1 GROUND_TRUTH_SEMANTICS 1
