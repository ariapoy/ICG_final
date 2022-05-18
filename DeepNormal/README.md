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