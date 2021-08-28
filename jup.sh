#!/bin/bash
source /home/yandex/DLW2021/davidhay/anaconda3/bin/activate
conda info --envs
conda install -y nb_conda_kernels
conda install -y ipykernel
jupyter notebook --no-browser --ip=0.0.0.0 --port 8888