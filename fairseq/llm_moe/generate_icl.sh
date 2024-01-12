





model_path="/mnt/nas/users/lsj/llm/output_dir_qwen/checkpoints/epoch0_iter300_loss2.334"
tokenizer_path="/mnt/nas/users/lsj/llm/pretrained_model/qwen_1b8"

deepspeed --num_gpus=2 --master_port=30000 DeepSpeed-Chat/generate_icl.py \
--model_path $model_path \
--tokenizer_path $tokenizer_path \
--src-lang "English" --tgt-lang "Chinese" \
--dataset_path /mnt/nas/users/lsj/moe/fairseq_data/raw_data/en-zh/test.en \
--out_path /mnt/nas/users/lsj/llm/res_dir/test.zh \
--deepspeed_config ds_config_inference.json \
--eval-batch-size 10 \
--is-moe