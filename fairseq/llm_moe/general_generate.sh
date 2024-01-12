

#model_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"
model_path="/mnt/nas/users/lsj/llm/output_dir_qwen/checkpoints/epoch0_iter2100_loss2.225"
tokenizer_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"

deepspeed --num_gpus=2 --master_port=30000 DeepSpeed-Chat/general_generate.py \
--model_path $model_path \
--tokenizer_path $tokenizer_path \
--dataset_path tmp.jsonl \
--out_path tmp.out \
--deepspeed_config ds_config_inference.json \
--eval-batch-size 25 \
--is-moe