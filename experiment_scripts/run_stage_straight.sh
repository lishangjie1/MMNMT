NUM_EXPERTS=32
max_tokens=3277
max_updates=100000
DATA="/mnt/nas/users/lsj/moe/moe_data/opus_dest/data-bin"
SAVE="/mnt/nas/users/lsj/moe/models/moe_model_init_encdec_cmr_straight_no_eom"
lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"

lang_dict="af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu"

# line > 500,000
high_pair="ar-en,bg-en,bn-en,bs-en,ca-en,cs-en,da-en,de-en,el-en,en-es,en-et,en-eu,en-fa,en-fi,en-fr,en-gl,en-he,en-hi,en-hr,en-hu,en-id,en-is,en-it,en-ja,en-ko,en-lt,en-lv,en-mg,en-mk,en-ml,en-ms,en-mt,en-nl,en-no,en-pl,en-pt,en-ro,en-ru,en-si,en-sk,en-sl,en-sq,en-sr,en-sv,en-th,en-tr,en-uk,en-ur,en-vi,en-zh,en-ar,en-bg,en-bn,en-bs,en-ca,en-cs,en-da,en-de,en-el,es-en,et-en,eu-en,fa-en,fi-en,fr-en,gl-en,he-en,hi-en,hr-en,hu-en,id-en,is-en,it-en,ja-en,ko-en,lt-en,lv-en,mg-en,mk-en,ml-en,ms-en,mt-en,nl-en,no-en,pl-en,pt-en,ro-en,ru-en,si-en,sk-en,sl-en,sq-en,sr-en,sv-en,th-en,tr-en,uk-en,ur-en,vi-en,zh-en"

# line <= 500,000
low_pair="af-en,am-en,as-en,az-en,be-en,br-en,cy-en,en-eo,en-fy,en-ga,en-gd,en-gu,en-ha,en-ig,en-ka,en-kk,en-km,en-kn,en-ku,en-ky,en-li,en-mr,en-my,en-nb,en-ne,en-nn,en-oc,en-or,en-pa,en-ps,en-rw,en-se,en-sh,en-ta,en-te,en-tg,en-tk,en-tt,en-ug,en-uz,en-wa,en-xh,en-yi,en-zu,en-af,en-am,en-as,en-az,en-be,en-br,en-cy,eo-en,fy-en,ga-en,gd-en,gu-en,ha-en,ig-en,ka-en,kk-en,km-en,kn-en,ku-en,ky-en,li-en,mr-en,my-en,nb-en,ne-en,nn-en,oc-en,or-en,pa-en,ps-en,rw-en,se-en,sh-en,ta-en,te-en,tg-en,tk-en,tt-en,ug-en,uz-en,wa-en,xh-en,yi-en,zu-en"


# debug
# export WORLD_SIZE=1
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=12345
# export RANK=0

# submit
echo $WORLD_SIZE # node num in DLC, need to multiply nproc
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK
export WORLD_SIZE=8

# stage one:
# only train moe parameter and use high resource languages
dense_model="/mnt/nas/users/lsj/moe/models/dense_model/bak/checkpoint_last.pt"

python train.py $DATA \
  --distributed-init-method "env://" --distributed-world-size $WORLD_SIZE --distributed-rank $RANK \
  --fp16-no-flatten-grads \
  --ddp-backend legacy_ddp --fp16 \
  --task translation_multi_simple_epoch \
  --langtoks-specs "main" \
  --langtoks "{\"main\":(\"src\", \"tgt\")}" \
  --sampling-method 'temperature' --sampling-temperature 5 \
  --langs ${lang_dict} --lang-pairs ${lang_pairs} \
  --enable-reservsed-directions-shared-datasets \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --share-all-embeddings \
  --encoder-layers 12 --decoder-layers 12 \
  --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 \
  --max-source-positions 1024 --max-target-positions 1024 \
  --skip-invalid-size-inputs-valid-test \
  --encoder-attention-heads 8 --decoder-attention-heads 8 \
  --moe-expert-count $NUM_EXPERTS --moe-freq 2 --capacity-factor 1.25 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.1 --moe-gate-loss-combine-method sum \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.5 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --max-update $max_updates \
  --dropout 0.3 --attention-dropout 0.1 \
  --max-tokens $max_tokens --update-freq 5 \
  --log-interval 10 \
  --save-interval-updates 3000 --save-dir $SAVE --keep-interval-updates 2 --no-epoch-checkpoints \
  --dataset-impl "mmap" \
  --record-a2a-perf-stats \
  --use-moe-pad-mask \
  --moe-batch-prioritized-routing \
  --symlink-best-and-last-checkpoints \
  --use-moe-cmr
