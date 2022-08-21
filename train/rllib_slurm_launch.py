# Usage:
# python train/rllib_slurm_launch.py \
#   --exp-name train_semexp \
#   --command "python train/train_semantic_exploration_policy.py --config_path submission/configs/ppo_custom_dataset_config.yaml" \
#   --load-env "conda activate habitat-challenge" \
#   --num-nodes 1 \
#   --num-gpus 8 \
#   --partition learnfair

import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

template_file = Path(__file__).resolve().parent / "rllib_slurm_template.sh"
JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)",
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
    )
    parser.add_argument(
        "--load-env",
        type=str,
        help="The script to load your environment ('module load cuda/10.1')",
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.",
    )
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(
        args.exp_name, time.strftime("%m%d-%H%M", time.localtime())
    )

    partition_option = (
        "#SBATCH --partition={}".format(args.partition) if args.partition else ""
    )

    # ===== Modify the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO " "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!",
    )

    # ===== Save the template script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    os.makedirs("slurm_logs", exist_ok=True)
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script file is at: <{}>. Log files are at: <{}>".format(
            script_file, "slurm_logs/{}.err/out".format(job_name)
        )
    )
    sys.exit(0)
