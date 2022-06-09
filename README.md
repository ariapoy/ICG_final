# README

This repo is for the final project of team 1, in NTU ICG 2022 spring.

## Env setting

Please check the version of CUDA, if you are using CUDA 10 please check [here](https://pytorch.org/get-started/locally/)

### Prerequirement

```shell
cd DeepNormal; git clone https://github.com/V-Sense/DeepNormals.git
cp *.py DeepNormals/
cd DeepNormals/; wget https://v-sense.scss.tcd.ie/Datasets/DeepNormalsModel.zip; unzip DeepNormalsModel.zip -d Net/
```

**Warning!**

- If you use `tensorflow>=1.5`, you need to change *model.py*.

```git
-       Network = tf.nn.l2_normalize(Network, dim = 3)
+       Network = tf.nn.l2_normalize(Network, axis = 3)
```

### Method 1 Python Venv/Virtualenv

```shell
make clean all
source ./venv/bin/activate
```
### Method 2 Anaconda

```shell
conda env create -f environment.yml
conda activate ICG
```

## Quick Start

Please see

- `harold.sh`

## Documents

Please see

- `traditional/README.md` for **combine sketch & tone**.
- `deepstyle/README.md` for **neural style transfer**.
- `DeepNormal/README.md` for **DeepNormal**.

If there is any problem, please feel free to send an email to ``d09944015@ntu.edu.tw''

## Authors

- Sheng-Wei Chen
- Yaxu Liu
- Poy Lu

## Reference

1. [Neural Transfer Using PyTorch](https://github.com/ariapoy/ICG_final.git)
2. [Combining sketch and tone for pencil drawing production](https://github.com/candycat1992/PencilDrawing)
3. [Deep Normals](https://github.com/V-Sense/DeepNormals)
