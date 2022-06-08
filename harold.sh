CUDA_VISIBLE_DEVICES="" # if no GPU
# harold
# option 1
# 1. get sketch, style transfer
# 1.1 combine sketch & tone
python pencilTrans.py --input_path harold.jpg -ks 16 -sw 0 -nd 8 -sks 3 -wg 1 -pt 3 -sd 2 -td 1 # harold-S.jpg, harold-s0.jpg
# 1.2 neural style transfer
python main.py --distance wgan-gp --samples 1024 --steps 1000 --alpha 0.1 --init-img content --style img-style/s1.jpg --content harold.jpg --out-dir harold --out_name harold-init=content-style=s1-content=harold-alpha=0.1-distance-wgan # harold-s1.png

# 2. normal map
python main.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --save_name harold-p1 --thining_it 10 --nb_grids 40 # harold-p1-normalmap.png

# 3. re-render
# 3.1 combine sketch & tone
python ICG_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path harold-p1-normalmap.png --color_path harold-s0.jpg --save_path harold-p1 # harold-p1/*
# 3.2 neural style transfer
python ICG_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path harold-p1-normalmap.png --color_path harold-s1.png --save_path harold-p1s1 # harold-p1s1/*

# option 2
# 1. get sketch
python pencilTrans.py --input_path harold.jpg -ks 16 -sw 0 -nd 8 -sks 3 -wg 1 -pt 3 -sd 2 -td 1 # harold-S.jpg

# 2. normal map
python main.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --save_name harold-p1 --thining_it 10 --nb_grids 40 # harold-p1-normalmap.png

# 3. re-render
python ICG_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path harold-p1-normalmap.png --color_path harold.jpg --save_path harold-p2 # harold-p2/*

# 4. style transfer
# 4.1 combine sketch & tone
python pencilTrans.py --input_path harold-9.jpg -ks 16 -sw 0 -nd 8 -sks 3 -wg 1 -pt 3 -sd 2 -td 1 # harold-9-s0.jpg

# 4.2 neural style transfer
python main.py --distance wgan-gp --samples 1024 --steps 1000 --alpha 0.1 --init-img content --style img-style/s1.jpg --content harold-9.jpg --out-dir harold --out_name harold-init=content-style=s1-content=harold9-alpha=0.1-distance-wgan # harold-9-s1.jpg