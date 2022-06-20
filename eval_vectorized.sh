#!/usr/bin/env bash

python eval_vectorized.py EVAL_VECTORIZED.specific_category "potted plant" AGENT.PLANNER.plant_dilation_selem_radius 12 EXP_NAME plant_dilation12
python eval_vectorized.py EVAL_VECTORIZED.specific_category tv AGENT.PLANNER.tv_dilation_selem_radius 10 EXP_NAME tv_dilation10
python eval_vectorized.py EVAL_VECTORIZED.specific_category tv AGENT.PLANNER.tv_dilation_selem_radius 12 EXP_NAME tv_dilation12
