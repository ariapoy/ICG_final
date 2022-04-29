filename="$1";
w=$2;
h=$3;
echo "$filename scale=${w}:${h}";
cd img-target/;
rm -rf ./$fileaname
mkdir -p ./$filename
ffmpeg -i $filename.mp4 -r 30 $filename/%3d.png
echo "Transform video to images."
cd ..;

outname="video-out-${filename}";
mkdir -p "$outname";
rm -rf img-style/*-resize.jpg
rm -rf img-target/*-resize.jpg

for target_name in `ls img-target/${filename}/`;
do
  target_idx=${target_name%.*};
  echo "img-target/${filename}/${target_idx}.png img-target/${filename}-${target_idx}-resize.jpg"
  ffmpeg -i "img-target/${filename}/${target_idx}.png" -vf scale=${w}:${h} "img-target/${filename}-${target_idx}-resize.jpg"
  ffmpeg -i "img-style/${filename}/0${target_idx}_8_1_3_0_0_1_0.5_1.png" -vf scale=${w}:${h} "img-style/${filename}-${target_idx}-resize.jpg"
  CUDA_VISIBLE_DEVICES=0 python3 main.py \
          --steps 1000 \
          --distance="wgan-gp" \
          --style="img-style/${filename}-${target_idx}-resize.jpg" \
          --content="img-target/${filename}-${target_idx}-resize.jpg" \
          --gray="True" \
          --device=cuda \
          --alpha=0.1 \
          --weight_S0 0.16666667\
          --weight_S1 0.16666667\
          --weight_S2 0.16666667\
          --weight_S3 0.16666667\
          --weight_S4 0.16666667\
          --weight_S5 0.16666667\
          --layer=5 \
          --init-img="img-style/${filename}-${target_idx}-resize.jpg" \
          --out-dir="./$outname" \
          --out_name="${filename}-${target_idx}";
done

cd ${outname};
rm -rf *.mp4
ffmpeg -i ${filename}-%3d.png -c:v libx264 -vf "fps=60,format=yuv420p" "${file_name}-deepstyle.mp4"

