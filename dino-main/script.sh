#!/bin/bash
FILES="/home/sylvain/Datasets/bdortho_roof_presence/*.jpeg"
for f in  $FILES
do
 b="$(basename -- $f)"
 c="${b%.*}"
 echo "Processing $c"
 d="/home/sylvain/Datasets/test_dino/tiny/$c"
 mkdir "$d"
 # do something on $f
 python visualize_attention.py --arch vit_tiny --pretrained_weights /home/sylvain/Datasets/saving_dino/checkpoint/checkpoint.pth --image_path $f --output_dir $d --patch_size=16
done