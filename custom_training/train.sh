
# export CUDA_VISIBLE_DEVICES=1
# set CUDA_LAUNCH_BLOCKING=1
# set TORCH_USE_CUDA_DSA=1
export WANDB_DISABLED=true

model_name="meta-llama/Llama-2-7b-chat-hf"
beta=0.1

for lr in 1e-4 1e-5 
do
    output_dir="${lr}"
    echo "Current Model: ${model_name}"
    echo "Current output_dir: ${output_dir}"
    python train.py \
        --model_name "${model_name}" \
        --beta "${beta}" \
        --lr "${lr}" \
        --output_dir "${output_dir}"
done
