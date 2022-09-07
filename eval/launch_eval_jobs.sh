#!/bin/bash

sbatch eval/eval_vectorized.sh EXP_NAME eval_standard
sbatch eval/eval_vectorized.sh EXP_NAME eval_goal_on_same_floor EVAL_VECTORIZED.goal_on_same_floor 1
sbatch eval/eval_vectorized.sh EXP_NAME eval_gt_semantics GROUND_TRUTH_SEMANTICS 1
sbatch eval/eval_vectorized.sh EXP_NAME eval_goal_on_same_floor_and_gt_semantics EVAL_VECTORIZED.goal_on_same_floor 1 GROUND_TRUTH_SEMANTICS 1
