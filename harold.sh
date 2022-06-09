CUDA_VISIBLE_DEVICES="" # if no GPU
# harold
# option 1
# 1. get sketch, style transfer
# 1.1 combine sketch & tone
cp img-input/harold.jpg traditional/; cd traditional;
python pencilTrans.py --input_path harold.jpg -ks 16 -sw 0 -nd 8 -sks 3 -wg 1 -pt 3 -sd 2 -td 1 # harold-S.jpg, harold-s0.jpg
cd ..;
# 1.2 neural style transfer
cp img-input/harold.jpg deepstyle/; cd deepstyle;
python main.py --distance wgan-gp --samples 1024 --steps 1000 --alpha 0.1 --init-img content --style img-style/s1.jpg --content harold.jpg --out-dir harold --out_name harold-init=content-style=s1-content=harold-alpha=0.1-distance-wgan # harold-s1.png
cp harold/harold-init=content-style=s1-content=harold-alpha=0.1-distance-wgan-s1.png harold-s1.png
cd ..;

# 2. normal map
cp traditional/harold-S.jpg DeepNormal/DeepNormals/; cd DeepNormal/DeepNormals/;
python main.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --save_name harold-p1 --thining_it 10 --nb_grids 40 # harold-p1-normalmap.png

# 3. re-render
# 3.1 combine sketch & tone
cp ../../traditional/harold-s0.jpg .;
python ICG_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path harold-p1-normalmap.png --color_path harold-s0.jpg --save_path harold-p1 # harold-p1/*
# 3.2 neural style transfer
cp ../../deepstyle/harold-s1.png .;
python ICG_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path harold-p1-normalmap.png --color_path harold-s1.png --save_path harold-p1s1 # harold-p1s1/*

# option 2
# 3. re-render on original image
cp ../../img-input/harold.jpg .;
python ICG_Rendering.py --lineart_path harold-S.jpg --mask_path harold-S.jpg --normal_path harold-p1-normalmap.png --color_path harold.jpg --save_path harold-p2 # harold-p2/*
cp harold-p2/9.png harold-p2-9.png
cd ../..;

# 4. style transfer
# 4.1 combine sketch & tone
cd traditional/; cp ../DeepNormal/DeepNormals/harold-p2-9.png .;
python pencilTrans.py --input_path harold-p2-9.png -ks 16 -sw 0 -nd 8 -sks 3 -wg 1 -pt 3 -sd 2 -td 1 # harold-p2-9-s0.jpg
cd ../..;

# 4.2 neural style transfer
cd deepstyle/; cp ../DeepNormal/DeepNormals/harold-p2-9.png .;
python main.py --distance wgan-gp --samples 1024 --steps 1000 --alpha 0.1 --init-img content --style img-style/s1.jpg --content harold-p2-9.png --out-dir harold --out_name harold-init=content-style=s1-content=harold9-alpha=0.1-distance-wgan # harold-9-s1.jpg
cp harold/harold-init=content-style=s1-content=harold9-alpha=0.1-distance-wgan-s1.png harold-p2-9-s1.png