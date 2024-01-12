
set -e
models=/mnt/nas/users/lsj/moe/models/moe_model_init-2_enc_stage_two_50k_50k_inverse/checkpoint_2_50000.pt
save_name="generate_dir"
capacity_factor=1 
DATA="/mnt/nas/users/lsj/moe/moe_data/opus_dest/data-bin"
src="en"
tgt="zh"
max_tokens=3000
resdir="$save_name"

spm_model="/mnt/nas/users/lsj/moe/moe_data/opus_dest/spm_data/spm.model"

mkdir -p $resdir

langs="af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu"


lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"
lang_pairs_to_generate=$lang_pairs
gpus=2

#
python -m torch.distributed.launch \
        --nproc_per_node=$gpus \
        --use_env \
        generate_multiple.py $DATA \
--task translation_multi_simple_epoch \
-s "$src" -t "$tgt" \
--lang-pairs $lang_pairs --lang-pairs-to-generate $lang_pairs_to_generate --langs "$langs" \
--path "$models" \
--max-tokens $max_tokens --beam 5 \
--bpe "sentencepiece" --sentencepiece-model $spm_model \
--results-path "$resdir" \
--langtoks-specs "main" --langtoks "{\"main\":(\"src\", \"tgt\")}" \
--dataset-impl "mmap" \
--skip-invalid-size-inputs-valid-test \
--is-moe \
--model-overrides "{'moe_eval_capacity_token_fraction':$capacity_factor}" \
# --enable-lang-ids \


# sacrebleu
bash calculate_bleu.sh $resdir "en" "fr"
