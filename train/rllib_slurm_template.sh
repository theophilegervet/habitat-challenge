#!/bin/bash
# shellcheck disable=SC2206
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!
${PARTITION_OPTION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}-%j.out
#SBATCH --error=slurm_logs/${JOB_NAME}-%j.err
#SBATCH --time=3-00:00:00
${GIVEN_NODE}
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=${NUM_NODES}
#SBATCH --exclusive
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${NUM_GPUS_PER_NODE}
#SBATCH --constraint=volta32gb

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate ${CONDA_ENV}
echo "(1) Load environment"
${LOAD_ENV}

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
echo "(2) Generate redis password"
${LOAD_ENV}
redis_password=$(uuidgen)
export redis_password
echo "Redis password: $redis_password"

echo "(3) Get node names"
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node names: $nodes"

echo "(4) Make redis address"
node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address)

if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detected a space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "(5) Starting head at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &
echo "(5) Command to launch $node_1 executed"
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1))  # Number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "(6) Starting worker $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --redis-password="$redis_password" --block &
  echo "(6) Command to launch worker $i at $node_i executed"
  sleep 5
done

# ===== Call your code below =====
${COMMAND_PLACEHOLDER}
