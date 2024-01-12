
set -e 

work_dir=$(realpath $(dirname $0))


# data_path="/mnt/nas/users/lsj/llm/data/qwen_data/bucket_0_text_document_qwen_15w"
# data_path="/mnt/nas/users/lsj/llm/data/new_ct_multidata/new_ct_multidata_text_document"
data_path="/mnt/nas/users/lsj/llm/data/multi_21b_4b/new_ct_multidata_text_document"

tokenizer_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"
#model_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"
model_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8" #,/mnt/nas/users/lsj/llm/output_dir_qwen_7b_ct_21b_4b/checkpoints/epoch0_iter6600_loss1.961" #,/mnt/nas/users/lsj/llm/output_dir_qwen_ct/checkpoints/epoch0_iter13200_loss1.977/"
output_path="/mnt/nas/users/lsj/llm/output_dir_qwen_ct_moe_21b_4b_2expert_continual_1"

# seed 1234 for qwen-ct lr=1e-4
# seed 1235 for moe-ct from qwen-ct  lr=5e-5
seed=1234
lr=1e-4
#lr=1e-4
train_params="\
    --deepspeed \
    --data_path $data_path \
    --model_name_or_path $model_path \
    --tokenizer_path $tokenizer_path \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 3 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed $seed \
    --zero_stage 2 \
    --gradient_checkpointing \
    --steps_per_save 6000 \
    --steps_per_print 1 \
    --output_dir $output_path \
    --max_seq_len 2047 \
    --is_moe \
    --gate_type continual_gate \
"
    # --lora_dim 8 \
    # --only_optimize_lora
    # --is_moe \
    # --gate_type fix_weight \
#--offload
#--is_moe \
    #--gate_type fix_weight
#--gradient_checkpointing \
    # --is_moe \
    # --gate_type discrete_gate
#    --offload \
# --gate_type discrete_gate \
# --is_moexw
    # --is_moe \
    # --gate_type discrete_gate
echo ${train_params}
# deepspeed --num_gpus 8 --num_nodes 7 DeepSpeed-Chat/train_from_scratch.py ${train_params}

echo $WORLD_SIZE # node num in DLC, need to multiply nproc
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK

# export CUDA_LAUNCH_BLOCKING=1

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --use_env \
        DeepSpeed-Chat/train_from_scratch.py ${train_params}


#deepspeed DeepSpeed-Chat/train_from_scratch.py ${train_params}


