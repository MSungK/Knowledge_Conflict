# export CUDA_VISIBLE_DEVICES=1
# set CUDA_LAUNCH_BLOCKING=1
# set TORCH_USE_CUDA_DSA=1
export WANDB_DISABLED=true
set -e

model_name="meta-llama/Llama-2-7b-chat-hf"

for lr in 1e-4 5e-4
do
    output_dir="SFT_lr:${lr}"
    echo "Current Model: ${model_name}"
    echo "Current output_dir: ${output_dir}"
    python SFTrain.py \
        --model_name "${model_name}" \
        --beta 1e-10 \
        --lr "${lr}" \
        --output_dir "${output_dir}"
done