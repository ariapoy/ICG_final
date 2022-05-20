# DeepNormal for Flexible Self-Shading

## Install Environment

```shell
git clone https://github.com/V-Sense/DeepNormals.git
cp ICG_Rendering.py DeepNormals
```

Then, you can follow the installation mannual of [DeepNormals project](https://github.com/V-Sense/DeepNormals).

**Warning!**

1. If you use `tensorflow>=1.5`, you need to change *model.py*.

```git
-       Network = tf.nn.l2_normalize(Network, dim = 3)
+       Network = tf.nn.l2_normalize(Network, axis = 3)
```

## Quick Start

1. Prepare the input of DeepNormal. You can use this [notebook on Colab](https://colab.research.google.com/drive/1c83qO4maKhVJ0Tg6VMkFa1-FDrIBNThu?usp=sharing).
    - Outline
    - Mask

2. Estimate normal vectors and flexible self-shading.

```shell
python main.py --lineart_path "Pepper/Line1.jpg" --mask "Pepper/Mask1.jpg" --save_path "canny"
python ICG_Rendering.py --lineart_path Pepper/Line1.jpg --mask_path Pepper/Mask1.jpg --normal_path RES/cannyNormal_Map-test.png --color_path Pepper/Color1.jpg
```

At the end, you'll see the nine images.

## Examples

### Harold


```shell
# step1 combine sketch and tone
cd ICG_final/traditional
python pencilTrans.py --input_path meinagano -sw 1 -sd 3

# step2 estimate normal vector
cd ICG_final/DeepNormal
python main.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --save_name harold

# step3 re-render

## 3.1 re-render with result of step1
python Interactive_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path RES/Normal_Map-harold.png --color_path harold-s0.jpg

## 3.2 re-render with original image
python Interactive_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path RES/Normal_Map-harold.png --color_path harold.jpg

# step4 style transfer
cd ICG_final/deepstyle
bash run-icg.sh harold 539 329
ffmpeg -i "out-harold/gen_harold-s1_isgray=True_init=content-_alpha=0.1_loss=wgan-gp.png" -vf scale=1240:758 "out-harold/harold-s1.png

```
