
src=$1
tgt=$2

# model1="moe_model_eom_0.75"
# model2="moe_model_eom_0.75_78000"
# model3="dense_model_0.5"
# model4="moe_model_stage_two_0.75"
# model5="moe_model_stage_two_0.75_encoder"
# model6="moe_model_stage_two_0.75_encoder_9000"
# model7="moe_model_stage_two_0.75_encoder_12000"
# model8="moe_model_stage_two_0.75_28k"
# echo "model: $model1"
# bash calculate_bleu.sh "../../generate_dir/$model1" $src $tgt
# echo "model: $model2"
# bash calculate_bleu.sh "../../generate_dir/$model2" $src $tgt
# echo "model: $model3"
# bash calculate_bleu.sh "../../generate_dir/$model3" $src $tgt
# echo "model: $model4"
# bash calculate_bleu.sh "../../generate_dir/$model4" $src $tgt
# echo "model: $model5"
# bash calculate_bleu.sh "../../generate_dir/$model5" $src $tgt

# echo "model: $model6"
# bash calculate_bleu.sh "../../generate_dir/$model6" $src $tgt

# echo "model: $model7"
# bash calculate_bleu.sh "../../generate_dir/$model7" $src $tgt
# echo "model: $model8"
# bash calculate_bleu.sh "../../generate_dir/$model8" $src $tgt



model1="dense_model_0.5"
model2="moe_model_base_1node_0.75_100k"
model3="moe_model_no_residual_init_encdec_mask_1node_0.75_100k"
model4="moe_model_mix_init_encdec_straight_0.75_93k"

echo "model: $model1"
bash calculate_bleu.sh "../../generate_dir/$model1" $src $tgt
echo "model: $model2"
bash calculate_bleu.sh "../../generate_dir/$model2" $src $tgt
echo "model: $model3"
bash calculate_bleu.sh "../../generate_dir/$model3" $src $tgt
echo "model: $model4"
bash calculate_bleu.sh "../../generate_dir/$model4" $src $tgt
