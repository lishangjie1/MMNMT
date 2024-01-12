
export CUDA_VISIBLE_DEVICES=0,1,2,3
models="/data/lsj/nfs/moe/moe_model3"
spm=""
input_path="/data/lsj/nfs/moe/moe_data"
src="fi"
tgt="en"
max_tokens=4096
generate_dir="/data/lsj/nfs/moe/moe_res"
lang_dict="en,fi" 
split="test"

python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=12345 \
generate.py $input_path \
--path "$models/checkpoint_last.pt"
--task translation_multi_simple_epoch \
--gen-subset  "$split" \
--beam 3 \
-s "$src" -t "$tgt" \
--bpe  "sentencepiece"  --sentencepiece-model  "$spm" \
--lang-pairs "$src-$tgt" --langs "$lang_dict" \
--langtoks-specs "main" --langtoks "{\"main\":(\"src\", \"tgt\")}" \
--max-tokens "$max_tokens" --scoring "sacrebleu" \
--results-path $generate_dir \
--is-moe  --fp16 --ddp-backend "fully_sharded" \
--skip-invalid-size-inputs-valid-test \
--enable-lang-ids 

cat $generate_dir/generate-$split.txt | tail -1

cat $generate_dir/generate-$split.txt | grep -P "^H" | sort -V | cut -f 3- > $generate_dir/decoding.txt

spm_decode --model $spm \
 --input $generate_dir/decoding.txt > $generate_dir/decoding.detok


echo "====BLEU score is ...."
raw_reference=$input_path/$split.$tgt
cat $generate_dir/decoding.detok | sacrebleu -b $raw_reference
