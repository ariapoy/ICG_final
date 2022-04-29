# Im2Pencil
Pytorch implementation of the CVPR19 [paper](https://arxiv.org/pdf/1903.08682.pdf) on controllable pencil illustration from photographs. 

## Getting started

- Linux
- NVIDIA GPU
- Pytorch 0.4.1
- MATLAB
- [Structured Edge Detection Toolbox](https://github.com/pdollar/edges) by Piotr Dollar 

```
git clone https://github.com/Yijunmaverick/Im2Pencil
cd Im2Pencil
```

## Preparation

- Download the pretrained models:

```
sh pretrained_models/download_models.sh
```

 - Extract the outline and tone image from the input photo (in MATLAB):
 
```
cd extract_edge_tone
Im2Pencil_get_edge_tone.m
```


## Testing

  - Test with different outline and shading styles：

```
python test.py  --outline_style 1  --shading_style 1
```
Outline style: 0 for `rough` and 1 for `clean`

Shading style: 0, 1, 2, 3 for `hatching`, `crosshatching`, `stippling`, and `blending` respectively

For other controllable parameters, check `options/test_options.py`

 - Or run all styles by

```bash
bash run.sh
```



