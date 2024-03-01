#!/bin/bash

python train/train_context.py "" --train-with train/addon_context.py --model resnet50 --pretrained --output ./output/classifier --no-prefetcher --num-classes 252 --epochs 100 --warmup-epochs 10 --cooldown-epochs 3 --batch-size 1 --sched cosine --lr 0.02 --amp -j 4 --checkpoint-hist 1 --experiment stage2

echo '>>> Done!!'

