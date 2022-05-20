filename="$1";
w=$2;
h=$3;
echo "$filename scale=${w}:${h}";

outname="out-${filename}";
mkdir -p "$outname";
rm -rf img-style/*-resize.jpg
rm -rf img-target/${filename}-resize.jpg
ffmpeg -i "img-target/${filename}.jpg" -vf scale=${w}:${h} "img-target/${filename}-resize.jpg"

style_name="s1";
ffmpeg -i "img-style/${style_name}.jpg" -vf scale=${w}:${h} "img-style/${style_name}-${filename}-resize.jpg" -y
alpha=0.1;
init="content";
loss="wgan-gp";
gray="True";
echo "${filename}-${style_name}_isgray=${gray}_init=${init}-${histmatch}_alpha=${alpha}_loss=${loss}";
CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --steps 1000 \
        --distance=${loss} \
        --style="img-style/${style_name}-${filename}-resize.jpg" \
        --content="img-target/${filename}-resize.jpg" \
        --gray=${gray} \
        --device=cuda \
        --alpha=${alpha} \
        --weight_S0 0.16666667\
        --weight_S1 0.16666667\
        --weight_S2 0.16666667\
        --weight_S3 0.16666667\
        --weight_S4 0.16666667\
        --weight_S5 0.16666667\
        --layer=5 \
        --init-img=${init} \
        --out-dir="./$outname" \
        --out_name="${filename}-${style_name}_isgray=${gray}_init=${init}-${histmatch}_alpha=${alpha}_loss=${loss}";
