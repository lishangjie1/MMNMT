#python=/path/to/python

# debug
# export WORLD_SIZE=1
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=12345
# export RANK=0

# submit
export WORLD_SIZE=16


encoder_layers=24
decoder_layers=24
embed_dim=2048
ffn_dim=8192
max_tokens=2000
num_experts=32
moe_freq=4

moe_args="--moe-gating-use-fp32 \
        --moe-second-expert-policy all \
        --moe-normalize-expert-grad sqrt_world_size \
        --criterion moe_cross_entropy \
        --moe-gate-loss-wt 0.01 \
        --moe-gate-loss-combine-method sum \
        --moe-batch-prioritized-routing \
        --use-moe-pad-mask \
        --moe-expert-count ${num_experts} \
        --fp16-no-flatten-grads \
        --moe-freq $moe_freq \
        --encoder-embed-dim $embed_dim \
        --decoder-embed-dim $embed_dim \
        --encoder-ffn-embed-dim $ffn_dim \
        --decoder-ffn-embed-dim $ffn_dim \
        --capacity-factor 1.0"

#--adam-betas '(0.9, 0.98)'
python train.py \
    --distributed-init-method "env://" --distributed-world-size $WORLD_SIZE --distributed-rank $RANK \
    --task dummy_mt \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --encoder-layers $encoder_layers --decoder-layers $decoder_layers \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion cross_entropy \
    --max-tokens $max_tokens \
    --ddp-backend fully_sharded \
    --fp16 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --no-epoch-checkpoints \
    --max-epoch 1 \
    --no-save \
    --disable-validation \
    --log-interval 10 \
    --log-format json \
    --record-a2a-perf-stats \
    --dataset-size 500000 \
    --update-freq 2 \
    $moe_args \
#     $scomoe_feat_args \
#     --no-save