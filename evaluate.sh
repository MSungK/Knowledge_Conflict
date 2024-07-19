base_path="custom_training/results/Llama-2-7b-chat-hf"

for check in 200 400 600 800 817
do
    python infer.py \
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
        --adapter_path ${check} \
        --data_path datasets/data/MRQANaturalQuestionsSPLIT-closedbookfiltered-corpus-counterfactual.json \
        --output_file ./${check}.json \
        --device auto \
        --num_shards 1 \
        --shard_id 0 \
        --add_eot_token \
        --prompt_template "instruction-based"
    
    mv ${check}.json NaturalQuestions_${check}.json

    python evaluate.py \
        --input_file ./${check}.json
done