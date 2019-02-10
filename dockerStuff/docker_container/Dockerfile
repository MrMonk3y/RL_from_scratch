FROM nvcr.io/nvidia/tensorflow:18.11-py3
RUN apt update && apt install -y xvfb python3-opengl
RUN apt-get update && apt-get install python3-tk -y
RUN apt-get install ffmpeg -y
RUN pip3 install gym
RUN pip3 --no-cache-dir install --no-dependencies git+https://github.com/keras-team/keras.git@${master}
RUN pip3 install pyvirtualdisplay
RUN pip3 install pyyaml
RUN pip3 install gym[atari]
RUN git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
RUN sh ~/.vim_runtime/install_awesome_vimrc.sh
WORKDIR /mnt/data/sjecklin/RL_from_scratch/src/connect4/
#ENTRYPOINT ["python"]

