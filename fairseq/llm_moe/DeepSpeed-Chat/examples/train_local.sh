# 这个例子是单机2卡的例子

num_gpus=2
deepspeedchat_root="/mnt/nas/users/funan.whr/workplace/factory/LLaMATraining/DeepSpeedExamples/applications/DeepSpeed-Chat"
output_dir="/mnt/nas/users/funan.whr/workplace/factory/LLaMATraining/tmp"

torchrun --nproc_per_node ${num_gpus} ${deepspeedchat_root}/train_sft.py \
    --deepspeed \
    --data_path "/mnt/nas/users/funan.whr/workplace/factory/LLaMATraining/alpaca_data_new.json" \
    --model_name_or_path "/mnt/nas/users/funan.whr/data/huggingface-models/opt-1.3b" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_seq_len 512 \
    --learning_rate 9.65e-6 \
    --weight_decay 0. \
    --num_train_epochs 16  \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage 3 \
    --output_dir ${output_dir} 2>&1 | tee deepspeed_chat.log