set -e
RESULT=$1
src=$2
tgt=$3

gen_dir=$RESULT/$src-$tgt

sudo chmod 777 $gen_dir
sudo chmod 777 $gen_dir/*

cat $gen_dir/generate-test.txt | grep -P "^D" | sort -V | cut -f 3- > $gen_dir/hyp
python recover_bpe.py $gen_dir/hyp $gen_dir/hyp.detok $src $tgt
cat $gen_dir/generate-test.txt | grep -P "^T" | sort -V | cut -f 2- > $gen_dir/ref
python recover_bpe.py $gen_dir/ref $gen_dir/ref.detok $src $tgt

if [ $tgt == 'zh' ]; then
    cat $gen_dir/hyp.detok | sacrebleu -w 3 -tok 'zh' -b $gen_dir/ref.detok
elif [ $tgt == 'ja' ]; then
    cat $gen_dir/hyp.detok | sacrebleu -w 3 -tok 'ja-mecab' -b $gen_dir/ref.detok
else
    cat $gen_dir/hyp.detok | sacrebleu -w 3 -tok '13a' -b $gen_dir/ref.detok
fi