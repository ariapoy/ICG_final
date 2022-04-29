for p in {0..1};
    do
    for s in {0..3};
        do
        CUDA_VISIBLE_DEVICES=0 python test.py  --outline_style $p  --shading_style $s
        done
    done