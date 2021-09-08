#! /bin/bash

#SBATCH --job-name=DLWSHP
#SBATCH --output=DLWSHP.out
#SBATCH --error=DLWSHP.err
#SBATCH --partition=studentbatch
#SBATCH --time=4000
#SBATCH --gpus=1
source /home/yandex/DLW2021/pelegv/anaconda3/bin/activate
python main.py --batch 3 --progress 4500 --load_vocab --checkpoint "/home/yandex/DLW2021/pelegv/checkpoint.pt" "/home/yandex/DLW2021/pelegv/coco/train2017" "/home/yandex/DLW2021/pelegv/coco/annotations/captions_train2017.json" 20