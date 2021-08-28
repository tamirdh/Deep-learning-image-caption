#! /bin/bash

#SBATCH --job-name=DLWSHP
#SBATCH --output=DLWSHP.out
#SBATCH --error=DLWSHP.err
#SBATCH --partition=studentbatch
#SBATCH --time=720
#SBATCH --gpus=1
source /home/yandex/DLW2021/davidhay/anaconda3/bin/activate
python main.py --batch 2  "/home/yandex/DLW2021/davidhay/coco/train2017" "/home/yandex/DLW2021/davidhay/coco/annotations/captions_train2017.json" 100