FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

ADD submission submission
RUN /bin/bash -c ". activate habitat; pip install -r submission/requirements.txt"
RUN /bin/bash -c ". activate habitat; pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html"
RUN /bin/bash -c ". activate habitat; git clone https://github.com/open-mmlab/mmdetection.git; pushd mmdetection; pip install -r requirements/build.txt; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'; pip install -v -e .; popd"

ADD agent.py agent.py
ADD submission.sh submission.sh
ADD challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]
