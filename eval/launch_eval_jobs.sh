#!/bin/bash

# Confusion matrix on HM3D, MP3D, Gibson
#DATASETS=("hm3d" "mp3d" "gibson")
#CHECKPOINTS=(
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_d3d73_00000_0_2022-10-03_11-40-18/checkpoint_001000/checkpoint-1000"
#  "/private/home/theop123/ray_results/ppo_train_mp3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_bcae0_00000_0_2022-10-01_08-00-07/checkpoint_000300/checkpoint-300"
#  "/private/home/theop123/ray_results/ppo_train_gibson_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_040bb_00000_0_2022-09-30_05-18-39/checkpoint_000900/checkpoint-900"
#)
#for i in ${!DATASETS[@]}; do
#  for j in ${!DATASETS[@]}; do
#    sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_${DATASETS[$i]}_config.yaml \
#      AGENT.POLICY.type semantic \
#      AGENT.POLICY.SEMANTIC.checkpoint_path ${CHECKPOINTS[$j]} \
#      EXP_NAME eval_${DATASETS[$j]}_on_${DATASETS[$i]}
#  done
#done

# Eval different HM3D checkpoints
#CHECKPOINT_NAMES=("700" "800" "900" "1100" "1300" "1500")
#CHECKPOINTS=(
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_bcfe1_00000_0_2022-09-30_03-00-40/checkpoint_000700/checkpoint-700"
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_bcfe1_00000_0_2022-09-30_03-00-40/checkpoint_000800/checkpoint-800"
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_d3d73_00000_0_2022-10-03_11-40-18/checkpoint_000900/checkpoint-900"
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_d3d73_00000_0_2022-10-03_11-40-18/checkpoint_001100/checkpoint-1100"
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_d3d73_00000_0_2022-10-03_11-40-18/checkpoint_001300/checkpoint-1300"
#  "/private/home/theop123/ray_results/ppo_train_hm3d_annotated_scenes/PPO_SemanticExplorationPolicyTrainingEnvWrapper_d3d73_00000_0_2022-10-03_11-40-18/checkpoint_001500/checkpoint-1500"
#)
#for i in ${!CHECKPOINTS[@]}; do
#  sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml \
#    AGENT.POLICY.type semantic \
#    AGENT.POLICY.SEMANTIC.checkpoint_path ${CHECKPOINTS[$i]} \
#    EXP_NAME eval_hm3d_ckpt_${CHECKPOINT_NAMES[$i]}
#done

# Ablation study on HM3D
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME neww_eval_standard
sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME neww_eval_goal_on_same_floor EVAL_VECTORIZED.goal_on_same_floor 1
sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME neww_eval_gt_semantics GROUND_TRUTH_SEMANTICS 1
sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME neww_eval_goal_on_same_floor_and_gt_semantics EVAL_VECTORIZED.goal_on_same_floor 1 GROUND_TRUTH_SEMANTICS 1
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME new_eval_2000_steps AGENT.max_steps 2000 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 2000
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME new_eval_goal_on_same_floor_and_gt_semantics_and_2000_steps EVAL_VECTORIZED.goal_on_same_floor 1 GROUND_TRUTH_SEMANTICS 1 AGENT.max_steps 2000 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 2000
#sbatch eval/eval_vectorized.sh --config_path=submission/configs/eval_hm3d_config.yaml EXP_NAME new_eval_SOME_failed_eps_no_goal PRINT_IMAGES 1 EVAL_VECTORIZED.specific_episodes 1 EVAL_VECTORIZED.record_videos 1
