#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'pc60 (pascal context)']
# method: ['semiovs_unimatch', 'semiovs_prevmatch']
# exp: just for specifying the 'save_path'
dataset='pascal'
method='semiovs_unimatch'
ood_id_path='unlabeled_coco_for_pascal'
exp='r101'
split=$3

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
unlabeled_ood_id_path=splits/ood/${ood_id_path}.txt

save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

export CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --unlabeled-ood-id-path $unlabeled_ood_id_path --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
