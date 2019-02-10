cd /mnt/data/sjecklin/RL_from_scratch
git pull

NV_GPU=0 nvidia-docker run --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /mnt/data/sjecklin/RL_from_scratch/src/connect4/:/mnt/data/sjecklin/RL_from_scratch/src/connect4/ -u $(id -u):$(id -g) openai-gym:0.1 
