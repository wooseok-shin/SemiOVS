#data
#├── custom_dataset
#│   ├── sample1.jpg
#│   ├── sample2.jpg
#      .
#      .

# sleep 1h

# dataset: ['pascal']
custom_dataset=$1
dataset=$2
config=configs/${dataset}.yaml
ckpt_path=$3
save_map=$4
img_size=$5     # Resizes while maintaining the aspect ratio, based on the larger value of h or w. (Pascal=500)

python inference.py --custom_dataset $custom_dataset --config $config --ckpt-path $ckpt_path \
                    --save-map $save_map --img-size $img_size