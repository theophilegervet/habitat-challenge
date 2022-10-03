#!/bin/bash

# Launch job on challenge dataset to make sure everything runs fine there
#python train/rllib_slurm_launch.py \
#    --exp-name ppo_overfit_challenge_hm3d \
#    --command "python train/train_semantic_exploration_policy.py --config_path submission/configs/ppo_overfit_challenge_hm3d_dataset_config.yaml" \
#    --load-env "" \
#    --num-nodes 1 \
#    --num-gpus 8 \
#    --partition learnfair \
#    --external-redis

# Overfit semantic exploration policy on one scene of HM3D, MP3D, Gibson
#for DATASET in hm3d mp3d gibson
#do
#  python train/rllib_slurm_launch.py \
#    --exp-name ppo_overfit_custom_${DATASET}_annotated_scenes \
#    --command "python train/train_semantic_exploration_policy.py --config_path submission/configs/ppo_overfit_custom_${DATASET}_annotated_scenes_dataset_config.yaml" \
#    --load-env "" \
#    --num-nodes 1 \
#    --num-gpus 8 \
#    --partition learnfair # \
#    #--external-redis
#done

# Train semantic exploration policy on HM3D, MP3D, Gibson with PPO and 1 node
for DATASET in hm3d mp3d gibson
do
  python train/rllib_slurm_launch.py \
    --exp-name ppo_train_custom_${DATASET}_annotated_scenes \
    --command "python train/train_semantic_exploration_policy.py --config_path submission/configs/ppo_train_custom_${DATASET}_annotated_scenes_dataset_config.yaml" \
    --load-env "" \
    --num-nodes 1 \
    --num-gpus 8 \
    --partition learnfair # \
    #--external-redis
done

# Train semantic exploration policy on HM3D, MP3D, Gibson with DDPPO and 8 nodes
#for DATASET in hm3d mp3d gibson
#do
#  python train/rllib_slurm_launch.py \
#    --exp-name ddppo_train_custom_${DATASET}_annotated_scenes \
#    --command "python train/train_semantic_exploration_policy.py --config_path submission/configs/ddppo_train_custom_${DATASET}_annotated_scenes_dataset_config.yaml" \
#    --load-env "" \
#    --num-nodes 8 \
#    --num-gpus 8 \
#    --partition learnfair \
#    --external-redis
#done
