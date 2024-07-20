base_path="custom_training/results/Llama-2-7b-chat-hf"
check=817

for adapter_path in 1e-4 1e-5
do
    python infer.py \
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
        --adapter_path ${adapter_path} \
        --checkpoint ${check} \
        --data_path datasets/data/MRQANaturalQuestionsSPLIT-closedbookfiltered-corpus-counterfactual.json \
        --output_file ./${adapter_path}.json \
        --device auto \
        --num_shards 1 \
        --shard_id 0 \
        --add_eot_token \
        --prompt_template "instruction-based"
    
    mv ${adapter_path}.json NaturalQuestions_${adapter_path}.json

    python evaluate.py \
        --input_file ./NaturalQuestions_${adapter_path}.json
done