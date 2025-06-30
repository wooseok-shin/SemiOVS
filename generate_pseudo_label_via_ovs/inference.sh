#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

#Pascal VOC
python inference.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-voc20 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco2voc_concepts_ensemble.json" \
 DATASETS.TEST \(\"voc_2012_test_background_sem_seg\"\,\) \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS pretrained/sed_model_large.pth \
 $opts


cat $output/eval-voc20/log.txt | grep copypaste