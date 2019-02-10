NV_GPU=0 nvidia-docker run --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /mnt/data/sjecklin:/mnt/data/sjecklin -u $(id -u):$(id -g) openai-gym:0.1 #gsurma_DQN.py --output /mnt/data/sjecklin/test_with_data_export
#NV_GPU=1 nvidia-docker run --rm -ti -v /home/sjecklin:/home/sjecklin -u $(id -u):$(id -g) openai-gym:0.1 python gsurma_DQN.py --output /mnt/data/test2
