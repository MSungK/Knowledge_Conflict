set -e

base_path="custom_training/results/Llama-2-7b-chat-hf"

for check in 40 80 120 160 200 240 280 320 360 440 480 520 560 600 640 680 720 760 817
do
    python evaluate.py \
        --input_file NaturalQuestions_${check}.json
done