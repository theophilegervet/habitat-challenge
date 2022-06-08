## Conda Dev Environment Setup

```
conda create -n habitat-challenge python=3.7 cmake=3.14.0
conda activate habitat-challenge
conda install habitat-sim-challenge-2022 headless -c conda-forge -c aihabitat
git clone --branch challenge-2022 https://github.com/facebookresearch/habitat-lab.git
pushd habitat-lab;pip install -r requirements.txt; python setup.py develop --all; popd
pip install -r submission/requirements.txt
python -m pip install git+https://github.com/facebookresearch/detectron2.git
```

## Make a Submission

```
# Build image
docker build . --file Objectnav.Dockerfile -t objectnav_submission

# Submit it to the leaderboard
evalai push objectnav_submission:latest --phase habitat-objectnav-minival-2022-1615
```
