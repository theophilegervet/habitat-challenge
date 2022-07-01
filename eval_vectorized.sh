#!/usr/bin/env bash

python eval_vectorized.py AGENT.POLICY.hint_in_frame 1 EXP_NAME jul1_hint_in_frame
python eval_vectorized.py AGENT.POLICY.type semantic EXP_NAME jul1_semantic_exploration
python eval_vectorized.py AGENT.panorama_start 1 EXP_NAME jul1_panorama_start
