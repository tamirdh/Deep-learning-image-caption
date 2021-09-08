#! /bin/bash

#SBATCH --job-name=Captions
#SBATCH --output=Captions.out
#SBATCH --error=Captions.err
#SBATCH --partition=studentbatch
#SBATCH --time=4000
#SBATCH --gpus=1
source /home/yandex/DLW2021/davidhay/anaconda3/bin/activate
python expirement.py --username davidhay --hidden 2096 --embed 512 --cnn v2 --rnn v4