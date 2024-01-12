
RESULT=$1
src=$2
tgt=$3

gen_dir=$RESULT/$src-$tgt

sudo chmod 777 $gen_dir
sudo chmod 777 $gen_dir/*

cat $gen_dir/generate-test.txt | grep -P "^D" | sort -V | cut -f 3- > $gen_dir/hyp
cat $gen_dir/generate-test.txt | grep -P "^T" | sort -V | cut -f 2- > $gen_dir/ref

cat $gen_dir/hyp | sacrebleu -w 3 $gen_dir/ref