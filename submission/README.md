## Conda Dev Environment Setup

```
conda create -n habitat-challenge python=3.7 cmake=3.14.0
conda activate habitat-challenge
conda install habitat-sim-challenge-2022 headless -c conda-forge -c aihabitat
git clone --branch challenge-2022 https://github.com/facebookresearch/habitat-lab.git
pushd habitat-lab;pip install -r requirements.txt; python setup.py develop --all; popd
pip install -r submission/requirements.txt

# Detectron2
python -m pip install git+https://github.com/facebookresearch/detectron2.git

# MMDetection - customize with your PyTorch and CUDA versions
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
git clone https://github.com/open-mmlab/mmdetection.git
pushd mmdetection; pip install -r requirements/build.txt; pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"; pip install -v -e .; popd

# Ray
pip install -U "ray[tune]"
pip install -U "ray[rllib]"
pip install -U "ray[air]"
conda install protobuf
```

## Code Structure

This repository contains modular components that can either be in the agent or environment, depending on what is most compute/communication-efficient for training or evaluation:
* `obs_preprocessor` preprocesses batches of observations into PyTorch tensors
* `semantic_map` contains the semantic map state and update
* `policy` selects high-level exploration/navigation goals
* `planner` selects low-level actions from high-level goals
* `visualizer` contains utilities to visualize observations, the semantic map, and goals

&nbsp;                  | `obs_preprocessor` | `semantic_map` | `policy` | `planner` | `visualizer` | agent entry point                                                     | env entry point
------------------------|--------------------|----------------|----------|-----------|--------------|-----------------------------------------------------------------------|----------------
Challenge inference     | agent              | agent          | agent    | agent     | agent        | `agent.py Agent.act()`                                                | `habitat.core.env Env.step()`
Vectorized evaluation   | env                | agent          | agent    | env       | env          | `agent.py Agent.prepare_planner_inputs()`                             | `vector_env/eval_env_wrapper.py EvalEnvWrapper.plan_and_step()`
Policy training with RL | env                | env            | agent    | env       | -            | `policy/semantic_exploration_policy.py SemanticExplorationPolicy.TODO()` | `vector_env/semexp_policy_training_env_wrapper.py SemexpPolicyTrainingEnvWrapper.step()`

## Make a Submission

```
# Build image
docker build . --file Objectnav.Dockerfile -t objectnav_submission

# Test locally
./test_locally_objectnav_rgbd.sh --docker-name objectnav_submission

# Submit it to the leaderboard
evalai push objectnav_submission:latest --phase habitat-objectnav-minival-2022-1615 --private
evalai push objectnav_submission:latest --phase habitat-objectnav-test-standard-2022-1615 --private
evalai push objectnav_submission:latest --phase habitat-objectnav-test-challenge-2022-1615 --private

# New leaderboard
evalai push objectnav_submission:latest --phase habitat-objectnav-test-standard-2022-696 --private
```
