FROM tensorflow/tensorflow:1.4.1-gpu-py3

RUN set -x && \
    pip3 install keras==2.1.3 && \
    pip3 install opencv-contrib-python==3.4.0.12 && \
    pip3 install keras-tqdm==2.0.1 && \
    pip3 install networkx==1.11 && \
    pip3 install hyperopt==0.1 && \
    pip3 install hyperas==0.4 && \
    apt-get update && \
    apt-get install -y libxext-dev libsm6 libxrender1 libfontconfig1

CMD ["/run_jupyter.sh", "--allow-root"]
