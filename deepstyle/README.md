# Deep Style

## Environment

CUDA GPU with 11GB memory 

Python      -- 3.6
PyTorch     -- 1.7.1
torchvision -- 0.8.2
Numpy       -- 1.19.4
Matplotlib  -- 3.3.3
Pillow      -- 8.0.1
TQDM        -- 4.54.1

## Usage Steps

1. Please download intput/ target image [here](https://drive.google.com/file/d/1uifU7Fe_AQKcQyMKeENyXGhNu5Z0lfpq/view?usp=sharing) and unzip it into `DIP\_final/deepstyle/img-target`. 
   Please download style image [here](https://drive.google.com/file/d/1uifU7Fe_AQKcQyMKeENyXGhNu5Z0lfpq/view?usp=sharing) and unzip it into `DIP\_final/deepstyle/img-style and unzip it into `DIP\_final/deepstyle/img-style`.
2. Use shell scripts `run-all.sh` (for all target images).
3. Move the results of **traditional algorithm** to  `img-style`. e.g. `cp -r ../traditional/output/video\_v1\_8\_1\_3\_0\_0\_1\_0.5\_1/ img-style/v1`
   Then `run-all-video.sh` (for all videos) to conduct all experiments.
4. You can use other style images, but the style images need to be as the same size as the content images.
   Remember to replace the style image name in shell scripts.
5. Use shell scripts `run.sh` (for the specific target images) and `run-video.sh` (for the specific video) to excute my program on the single experiment.
6. You can also use `main.py` to training, please use `python3 main.py --help` to see the flags that I support.

## Note of `run-all.sh`

- Update "${filename}" "${widht}" "${height}"

```shell
# run all experiments
# h * w = 182040 for 8 GB memory
# h * w = 273060 for 12 GB memory

# at luhome (8 GB memory)
# bash run.sh "Lift\_Poy-Lu\_resized" 370 493 # 454 605
bash run.sh "101" 339 452; # 339 452
```


### Note of `run.sh`

List of essential arguments for experiments:
1. `style\_idx in {1..6}`: style image with file name "s1, s2, s3, ..." and so on.
   - "s5" is the result of **sketch and tone** (traditional method).
2. `init in "img-style/s5-${filename}-resize.jpg" "content" "random"`: initial matrix.
3. `loss in "wgan-gp" "gram"`: loss function.
4. `gray in "True"`: gray (True) or RGB (False).

If there is any question, please feel free to send an email to ``d09944015@ntu.edu.tw''.

D09944003 Sheng-Wei Chen
D09944015 Po-Yi Lu

