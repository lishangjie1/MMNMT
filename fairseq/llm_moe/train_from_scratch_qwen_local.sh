
set -e 

work_dir=$(realpath $(dirname $0))


# data_path="/mnt/nas/users/lsj/llm/data/qwen_data/bucket_0_text_document_qwen_15w"
# data_path="/mnt/nas/users/lsj/llm/data/new_ct_multidata/new_ct_multidata_text_document"
data_path="/mnt/nas/users/lsj/llm/data/multi_21b_4b/new_ct_multidata_text_document"

tokenizer_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_7b"
#model_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"
model_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_7b" #,/mnt/nas/users/lsj/llm/output_dir_qwen_ct/checkpoints/epoch0_iter13200_loss1.977/"
output_path="/mnt/nas/users/lsj/llm/output_dir_qwen_ct_7b"

# seed 1234 for qwen-ct lr=1e-4
# seed 1235 for moe-ct from qwen-ct  lr=5e-5
seed=1234
lr=1e-4

train_params="\
    --deepspeed \
    --data_path $data_path \
    --model_name_or_path $model_path \
    --tokenizer_path $tokenizer_path \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed $seed \
    --gradient_checkpointing \
    --offload \
    --zero_stage 2 \
    --steps_per_save 600 \
    --steps_per_print 1 \
    --output_dir $output_path \
    --max_seq_len 2047 \
"
#--is_moe \
 #   --gate_type fix_weight
    # --is_moe \
    # --gate_type discrete_gate
#    --offload \
# --gate_type discrete_gate \
# --is_moe
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
        --nproc_per_node=1 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=25000 \
        --use_env \
        DeepSpeed-Chat/train_from_scratch.py ${train_params}


#deepspeed DeepSpeed-Chat/train_from_scratch.py ${train_params}


