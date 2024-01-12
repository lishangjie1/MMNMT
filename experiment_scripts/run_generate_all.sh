
set -e
# submit
echo $WORLD_SIZE # node num in DLC, need to multiply nproc
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK
export WORLD_SIZE=8

# debug
# export WORLD_SIZE=1
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=12345
# export RANK=0

model_dir="/mnt/nas/users/lsj/moe/models"


model_name="moe_model_mix_init_encdec_straight"
model="$model_dir/$model_name/checkpoint_4_99000.pt"
model_save="/mnt/nas/users/lsj/moe/generate_dir/$model_name"

capacity=0.75
bash run_generate_multi.sh $model ${model_save}_${capacity}_100k $capacity