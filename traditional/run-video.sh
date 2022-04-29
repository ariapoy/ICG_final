#!/bin/bash

root=$1
file_dir=`dirname $root`
file_name="${root##*/}"
img_dir="$file_dir/${file_name%.*}"
mkdir -p ./$img_dir

ffmpeg -i $root -r 30 $img_dir/%4d.png
wait

ks=8
sw=1
sks=3
wg=0
pt=0
sd=1
td=0.5
rgb=1
save_dir="output/video_${file_name%.*}_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}"
mkdir -p ./$save_dir

task(){
for f in `ls ./$img_dir`;
do
    init_cmd="python pencilTrans.py --input_path $img_dir/$f -nd 8"
    cmd="$init_cmd -ks $ks"
    cmd="$cmd -sw $sw"
    cmd="$cmd -sks $sks"
    cmd="$cmd -wg $wg"
    cmd="$cmd -pt $pt"
    cmd="$cmd -sd $sd"
    cmd="$cmd -td $td"
    if [ "$rgb" == "1" ]; then
        cmd="$cmd --rgb"
        echo "$cmd --save_path $save_dir/${f%.*}_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}.png"
    else
        echo "$cmd --save_path $save_dir/${f%.*}_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}.png"
    fi
done
}

#task

task | xargs -0 -d '\n' -P 8 -I {} sh -c {}
wait
rm -rf output/${file_name%.*}_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}.mp4
ffmpeg -i $save_dir/%4d_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}.png -c:v libx264 -vf "fps=60,format=yuv420p" output/${file_name%.*}_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}.mp4
mv output/${file_name%.*}_${ks}_${sw}_${sks}_${wg}_${pt}_${sd}_${td}_${rgb}.mp4 output/${file_name%.*}-traditional.mp4

