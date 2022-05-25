# README

This repo is for the final project of team 1, in NTU ICG 2022 spring.

## Env setting

Please check the version of CUDA, if you are using CUDA 10 please check [here](https://pytorch.org/get-started/locally/)

```
make clean all
source ./venv/bin/activate
```

For **Im2Pencil** enviornment setting, please see `im2pencil/README.md`.

## Quick Start

```shell
# 1 comb sketch & tone
cd traditional/
python pencilTrans.py --input_path teapot3.png -sw 1 -sd 3
# 2 style transfer
cd deepstyle/
python main.py --style img-style/s1.jpg --content teapot3.png --out-dir teapot3 --out_name teapot3-s1 --alpha 0.1
# 3 DeepNormal
cd DeepNormal/DeepNormals/
python main.py --lineart_path teapot3-S.jpg --mask_path teapot-S.jpg --save_name teapot-S
```

## Quick Start

Please see

- `traditional/README.md` for traditional algorithm.
- `deepstyle/README.md` for deep style transfer.
- `im2pencil/README.md` for **Im2Pencil**.

If there is any problem, please feel free to send an email to ``d09944015@ntu.edu.tw''

## Authors

- Sheng-Wei Chen
- Yaxu Liu
- Poy Lu
- Cheng-Kun Yang

## Reference

1. https://github.com/milesial/Pytorch-UNet

