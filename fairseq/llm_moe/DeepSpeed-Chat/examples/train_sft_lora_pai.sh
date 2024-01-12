
set -e

odpscmd="/mnt/nas/users/funan.whr/odps/bin/odpscmd"
work_dir=$(realpath $(dirname $0))
oss_auth_token="?role_arn=acs:ram::1171142387015685:role/alitranxpuboss&host=cn-zhangjiakou.oss.aliyuncs.com"


data_path="oss://alitranx-public/funan.whr/data/huggingface_datasets/multi_alpaca_12000.jsonl"
model_oss_path="oss://alitranx-public/funan.whr/llm_models/llama-13b"
output_oss_path="oss://alitranx-public/funan.whr/tmp/dschat_llama13b_sft_test_v100"

# oss_bucket_0: model_oss_path
# oss_bucket_1: data_path
# oss_bucket_2: output_oss_path
Dbuckets="oss://alitranx-public/funan.whr/llm_models/${oss_auth_token},oss://alitranx-public/funan.whr/data/huggingface_datasets/${oss_auth_token},oss://alitranx-public/funan.whr/tmp/${oss_auth_token}"

model_smartcache_path="/data/oss_bucket_0/llama-13b"
input_smartcache_path="/data/oss_bucket_1/multi_alpaca_12000.jsonl"
output_smartcache_path="/data/oss_bucket_2/dschat_llama13b_sft_test_v100"

deepspeedchat_root="/mnt/nas/users/funan.whr/workplace/factory/LLaMATraining/DeepSpeedExamples/applications/DeepSpeed-Chat"


rm -rf ${work_dir}/code.tar.gz
tar zcvf ${work_dir}/code.tar.gz -C ${work_dir}/DeepSpeedExamples/applications/DeepSpeed-Chat .


train_params="\
    --deepspeed \
    --data_path ${input_smartcache_path} \
    --model_name_or_path ${model_smartcache_path} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --max_seq_len 512 \
    --learning_rate 9.65e-6 \
    --weight_decay 0. \
    --num_train_epochs 1  \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage 3 \
    --steps_per_save 500 \
    --steps_per_print 1 \
    --offload \
    --lora_dim 128 \
    --lora_module_name transformer.h \
    --only_optimize_lora \
    --wandb_enable \
    --wandb_team whrtranx \
    --wandb_project test_wandb_pai \
    --wandb_key 713b66df7cf92888474974549f7fdfc160b08e2a \
    --wandb_group dschat_llama13b_sft_test_v100 \
    --output_dir ${output_smartcache_path} \
"
echo ${train_params}


odpscmd_cmd="
    set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
    pai -name pytorch112z -project algo_public
    -Dscript=\"file://${work_dir}/code.tar.gz\"
    -DentryFile=\"-m torch.distributed.launch --use_env train_sft.py $(echo ${train_params}) \"
    -Dbuckets='${Dbuckets}'
    -Dcluster='{\"worker\": {\"gpu\": 800, \"cpu\": 3200, \"memory\": 400000}}'
    -DworkerCount=1;
"

echo ${odpscmd_cmd}

${odpscmd} -e "${odpscmd_cmd}" 2>&1 | tee train_sft_1node.log