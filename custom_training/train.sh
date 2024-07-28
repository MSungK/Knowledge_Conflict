# export CUDA_VISIBLE_DEVICES=1
# set CUDA_LAUNCH_BLOCKING=1
# set TORCH_USE_CUDA_DSA=1
export WANDB_DISABLED=true
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
set -e

model_name="meta-llama/Llama-2-7b-hf"
lora_alpha=32
max_grad_norm=0.7
weight_decay=0.01
lr=5e-4
beta=0.5

output_dir="KC/"
echo "Current Working: ${output_dir}"
python train.py \
    --model_name ${model_name} \
    --beta ${beta} \
    --lr ${lr} \
    --output_dir ${output_dir} \
    --lora_alpha ${lora_alpha} \
    --weight_decay ${weight_decay} \
    --max_grad_norm ${max_grad_norm} 


# model_name="meta-llama/Llama-2-7b-hf"
# lora_alpha=16
# max_grad_norm=0.7
# weight_decay=0.01
# lr=5e-4

output_dir="SFT/"
echo "Current Working: ${output_dir}"
python SFTrain.py \
    --model_name ${model_name} \
    --beta 1e-10 \
    --lr ${lr} \
    --output_dir ${output_dir} \
    --lora_alpha ${lora_alpha} \
    --weight_decay ${weight_decay} \
    --max_grad_norm ${max_grad_norm} 