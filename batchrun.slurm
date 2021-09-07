#! /bin/bash

#SBATCH --job-name=DLWSHP
#SBATCH --output=DLWSHP.out
#SBATCH --error=DLWSHP.err
#SBATCH --partition=studentbatch
#SBATCH --time=960
#SBATCH --gpus=1
source /home/yandex/DLW2021/pelegv/anaconda3/bin/activate
python main.py --batch 3 "/home/yandex/DLW2021/pelegv/coco/val2017" "/home/yandex/DLW2021/pelegv/coco/annotations/captions_val2017.json" 13