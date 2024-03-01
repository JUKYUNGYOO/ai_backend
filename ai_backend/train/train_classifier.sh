#!/bin/bash

trap '' INT

TEMP_DIR=$(mktemp -u -d -q ./temp_imagefolder_XXXXX)

(trap - INT; python train/train_classifier.py $TEMP_DIR --train-with train/addon_classifier.py --model resnet50 --pretrained --output ./output/classifier --experiment stage1 --no-prefetcher --num-classes 252 --epochs 50 --warmup-epochs 3 --cooldown-epochs 3 --batch-size 1 --sched cosine --lr 1.6 --amp -j 4 --checkpoint-hist 1 )

if [ -d $TEMP_DIR ] 
then
    echo '>>> Removing temporary directory:' $TEMP_DIR
    rm -rf $TEMP_DIR
fi

echo '>>> Done!!'