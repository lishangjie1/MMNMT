
set -e 

work_dir=$(realpath $(dirname $0))


data_path="/mnt/nas/users/lsj/llm/data/qwen_data/bucket_0_text_document_qwen_15w"
# data_path="/mnt/nas/users/lsj/llm/data/new_ct_multidata/new_ct_multidata_text_document"
model_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"
output_path="/mnt/nas/users/lsj/llm/output_dir_qwen_ct_moe_continual_gate"

train_params="\
    --deepspeed \
    --data_path $data_path \
    --model_name_or_path $model_path \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 5 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --steps_per_save 300 \
    --steps_per_print 1 \
    --output_dir $output_path \
    --max_seq_len 2047 \
    --offload \
    --is_moe \
    --gate_type continual_gate \
"
#    --offload \
# --is_moe
echo ${train_params}
# deepspeed --num_gpus 8 --num_nodes 7 DeepSpeed-Chat/train_from_scratch.py ${train_params}

echo $WORLD_SIZE # node num in DLC, need to multiply nproc
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK


python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --use_env \
        DeepSpeed-Chat/train_from_scratch.py ${train_params}


#deepspeed DeepSpeed-Chat/train_from_scratch.py ${train_params}


