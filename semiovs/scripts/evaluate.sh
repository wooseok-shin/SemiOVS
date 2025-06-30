#!/bin/bash

# modify the dataset argument if you want to try other datasets.
# dataset: ['pascal']
dataset=$1
config=configs/${dataset}.yaml
ckpt_path=$2
save_map=$3

python evaluate.py --config $config --ckpt-path $ckpt_path --save-map $save_map