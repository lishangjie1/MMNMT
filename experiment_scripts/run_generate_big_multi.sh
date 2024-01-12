
set -e
models=$1
save_name=$2
capacity_factor=$3
DATA="/mnt/nas/users/lsj/moe/fairseq_data/data-bin/shard-0"
src="en"
tgt="zh"
max_tokens=3000
resdir="$save_name"

spm_model="/mnt/nas/users/lsj/moe/fairseq_data/spm_data/spiece.model"

mkdir -p $resdir

langs="af,ar,bg,bn,cs,da,de,el,en,es,et,fa,fi,fr,he,hi,hr,hu,id,is,it,ja,kk,ko,ky,lo,lt,ms,my,nl,pl,pt,ro,ru,sk,sl,sr,sv,ta,tg,th,tk,tr,uk,uz,vi,zh"
#lang_pairs="ar-en,de-en,en-ar,en-de,en-es,en-fr,en-he,en-id,en-it,en-ja,en-ko,en-nl,en-pl,en-pt,en-ru,en-th,en-tr,en-vi,es-en,fr-en,he-en,id-en,it-en,ja-en,ko-en,nl-en,pl-en,pt-en,ru-en,th-en,tr-en,vi-en"
lang_pairs="fr-en" #"en-ar,ar-en,en-de,de-en,en-es,es-en,en-fr,fr-en,en-he,he-en,en-id,id-en,en-it,it-en,en-ja,ja-en,en-ko,ko-en,en-nl,nl-en,en-pl,pl-en,en-pt,pt-en,en-ru,ru-en,en-th,th-en,en-tr,tr-en,en-vi,vi-en,af-zh,ar-zh,bg-zh,bn-zh,cs-zh,da-zh,de-zh,el-zh,en-zh,es-zh,et-zh,fa-zh,fi-zh,fr-zh,he-zh,hi-zh,hr-zh,hu-zh,id-zh,is-zh,it-zh,ja-zh,kk-zh,ko-zh,ky-zh,lo-zh,lt-zh,ms-zh,my-zh,nl-zh,pl-zh,pt-zh,ro-zh,ru-zh,sk-zh,sl-zh,sr-zh,sv-zh,ta-zh,tg-zh,th-zh,tk-zh,tr-zh,uk-zh,uz-zh,vi-zh,zh-af,zh-ar,zh-bg,zh-bn,zh-cs,zh-da,zh-de,zh-el,zh-en,zh-es,zh-et,zh-fa,zh-fi,zh-fr,zh-he,zh-hi,zh-hr,zh-hu,zh-id,zh-is,zh-it,zh-ja,zh-kk,zh-ko,zh-ky,zh-lo,zh-lt,zh-ms,zh-my,zh-nl,zh-pl,zh-pt,zh-ro,zh-ru,zh-sk,zh-sl,zh-sr,zh-sv,zh-ta,zh-tg,zh-th,zh-tk,zh-tr,zh-uk,zh-uz,zh-vi"
lang_pairs_to_generate=$lang_pairs

python generate_multiple.py $DATA \
--distributed-init-method "env://" --distributed-world-size $WORLD_SIZE --distributed-rank $RANK \
--task translation_multi_simple_epoch \
-s "$src" -t "$tgt" \
--lang-pairs $lang_pairs --lang-pairs-to-generate $lang_pairs_to_generate --langs "$langs" \
--path "$models" \
--max-tokens $max_tokens --beam 5 \
--bpe "sentencepiece" --sentencepiece-model $spm_model \
--results-path "$resdir" \
--langtoks-specs "main" --langtoks "{\"main\":(\"src\", \"tgt\")}" \
--dataset-impl "mmap" \
--skip-invalid-size-inputs-valid-test
