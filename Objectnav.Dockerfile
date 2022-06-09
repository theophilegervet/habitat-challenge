FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

ADD submission submission
RUN /bin/bash -c ". activate habitat; pip install scikit-image==0.15.0; pip install scikit-fmm; pip install torchvision; python -m pip install git+https://github.com/facebookresearch/detectron2.git"

ADD agent.py agent.py
ADD submission.sh submission.sh
ADD challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]
