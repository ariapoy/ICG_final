filename="$1";
w=$2;
h=$3;
echo "$filename scale=${w}:${h}";

outname="out-${filename}";
mkdir -p "$outname";
rm -rf img-style/*-resize.jpg
rm -rf img-target/${filename}-resize.jpg
ffmpeg -i "img-target/${filename}.jpg" -vf scale=${w}:${h} "img-target/${filename}-resize.jpg"
ffmpeg -i "img-style/${filename}/s5.jpg" -vf scale=${w}:${h} "img-style/s5-${filename}-resize.jpg"

for style_idx in {1..6};
do
  ffmpeg -i "img-style/${filename}/s${style_idx}.jpg" -vf scale=${w}:${h} "img-style/s${style_idx}-${filename}-resize.jpg" -y
  for alpha in 0.1;
  do
    for init in "img-style/s5-${filename}-resize.jpg" "content";
    do
        for loss in "wgan-gp" "gram";
        do
        for gray in "True";
          do
          echo "${filename}-${style_idx}_isgray=${gray}_init=${init}-${histmatch}_alpha=${alpha}_loss=${loss}";
          CUDA_VISIBLE_DEVICES=0 python3 main.py \
                  --steps 1000 \
                  --distance=${loss} \
                  --style="img-style/s${style_idx}-${filename}-resize.jpg" \
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
                  --out_name="${filename}-${style_idx}_isgray=${gray}_init=${init}-${histmatch}_alpha=${alpha}_loss=${loss}";
          done
        done
    done
  done
done
