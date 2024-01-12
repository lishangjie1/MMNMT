# 这个例子可以在PAI Studio集群上启动llama-13b的训练
# 请将output_oss_path换成自己的oss路径，避免冲突
set -e

odpscmd="/mnt/nas/users/funan.whr/odps/bin/odpscmd"

work_dir=$(realpath $(dirname $0))


input_data="oss://alitranx-public/funan.whr/alpaca_data/alpaca_data_new.json"
model_oss_path="oss://alitranx-public/funan.whr/llm_models/llama-13b"
output_oss_path="oss://alitranx-public/funan.whr/tmp/deepspeedchat_xxxxxx"


deepspeedchat_root="/mnt/nas/users/funan.whr/workplace/factory/LLaMATraining/DeepSpeedExamples/applications/DeepSpeed-Chat"
output_dir="/mnt/nas/users/funan.whr/workplace/factory/LLaMATraining/tmp"

oss_auth_token="?role_arn=acs:ram::1171142387015685:role/alitranxpuboss&host=cn-zhangjiakou.oss.aliyuncs.com"


rm -rf ${work_dir}/code.tar.gz
tar zcvf ${work_dir}/code.tar.gz -C ${work_dir}/DeepSpeedExamples/applications/DeepSpeed-Chat .

model_smartcache_path="/data/oss_bucket_0/llama-13b"
input_smartcache_path="/data/oss_bucket_1/alpaca_data_new.json"
output_smartcache_path="/data/oss_bucket_2/deepspeedchat_tmp"

# cd ${deepspeedchat_root}

train_params="\
    --deepspeed \
    --data_path ${input_smartcache_path} \
    --model_name_or_path ${model_smartcache_path} \
    --per_device_train_batch_size 8 \
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
    --offload \
    --output_dir ${output_smartcache_path} \
"

echo ${train_params}


odpscmd_cmd="
    set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
    pai -name pytorch112z -project algo_public
    -Dscript=\"file://${work_dir}/code.tar.gz\"
    -DentryFile=\"-m torch.distributed.launch --use_env train_sft.py $(echo ${train_params}) \"
    -Dbuckets='oss://alitranx-public/funan.whr/llm_models/${oss_auth_token},oss://alitranx-public/funan.whr/alpaca_data/${oss_auth_token},oss://alitranx-public/funan.whr/tmp/${oss_auth_token}'
    -Dcluster='{\"worker\": {\"gpu\": 800, \"cpu\": 3200, \"memory\": 400000}}'
    -DworkerCount=1;
"

echo ${odpscmd_cmd}

${odpscmd} -e "${odpscmd_cmd}" 2>&1 | tee infer.log


