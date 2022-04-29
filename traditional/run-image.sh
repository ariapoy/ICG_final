#!/bin/bash

root=$1
nd=8
save_dir="output/"
mkdir -p $save_dir

task(){
for f in `ls $root`;
do
    if file "$root/$f" | grep -qE 'image|bitmap'; then
        init_cmd="python pencilTrans.py --input_path $root/$f -nd 8"
        for ks in 8 16 32;
        do
            for sw in 1 2;
            do
                for sks in 3;
                do
                    for wg in 0 1 2;
                    do
                        for pt in `seq 0 4`;
                        do
                            for sd in 1;
                            do
                                for td in 0.5 1 2;
                                do
                                    for rgb in 0 1;
                                    do
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
                                done
                            done
                        done
                    done
                done
            done
        done
    fi
done
}

#task

task | xargs -0 -d '\n' -P 8 -I {} sh -c {}
