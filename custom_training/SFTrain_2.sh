# export CUDA_VISIBLE_DEVICES=1
# set CUDA_LAUNCH_BLOCKING=1
# set TORCH_USE_CUDA_DSA=1
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="2,3"
set -e

model_name="meta-llama/Llama-2-7b-chat-hf"
lr=5e-4
lora_alpha=32

for weight_decay in 0.1 0.01
do
    output_dir="SFT_alpha:${alpha}_wd:${weight_decay}"
    python SFTrain.py \
        --model_name ${model_name} \
        --beta 1e-10 \
        --lr ${lr} \
        --output_dir ${output_dir} \
        --lora_alpha ${lora_alpha} \
        --weight_decay ${weight_decay} 
done