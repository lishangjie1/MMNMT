alitranx_nas="/root/E-funan.whr-170101/nas"

deepspeedchat_root="${alitranx_nas}/users/funan.whr/workplace/factory/LLaMATraining/DeepSpeedExamples/applications/DeepSpeed-Chat"
output_dir="${alitranx_nas}/users/funan.whr/workplace/factory/LLaMATraining/alitranx_sft"
# /mnt/nas/users/funan.whr/data/huggingface-models/alitranx/pretrain_v2/ct_from_40000_step_22350
mkdir -p ${output_dir}

pip install -r ${deepspeedchat_root}/requirements.txt

torchrun ${deepspeedchat_root}/train_sft.py \
    --deepspeed \
    --data_path "${alitranx_nas}/users/funan.whr/data/huggingface-datasets/sft/multi_alpaca_12000.jsonl" \
    --model_name_or_path "${alitranx_nas}/users/funan.whr/data/huggingface-models/alitranx/pretrain_v2/ct_from_40000_step_22350" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --max_seq_len 512 \
    --learning_rate 9.65e-6 \
    --weight_decay 0. \
    --num_train_epochs 5  \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage 3 \
    --lora_dim 128 \
    --lora_module_name "transformer.h" \
    --only_optimize_lora \
    --steps_per_print 10 \
    --steps_per_save 5000 \
    --output_dir ${output_dir} 2>&1

# | tee ${output_dir}/deepspeed_chat.log