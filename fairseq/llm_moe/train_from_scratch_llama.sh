
set -e 

work_dir=$(realpath $(dirname $0))


data_path="/mnt/nas/users/lsj/llama/data/ct_train_zhen_llama_text"
model_path="/mnt/nas/users/lsj/llama/pretrained_model/llama_7b"
output_path="/mnt/nas/users/lsj/llama/output_dir"

train_params="\
    --deepspeed \
    --data_path $data_path \
    --model_name_or_path $model_path\
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --steps_per_save 1000 \
    --steps_per_print 1 \
    --offload \
    --output_dir $output_path\
"
#     \
echo ${train_params}


deepspeed DeepSpeed-Chat/train_from_scratch.py ${train_params}

