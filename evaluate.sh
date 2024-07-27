base_path="custom_training"  # Shoud specify the path in python scirpt(infer.py)
check=1635
adapter_path=KC


python infer.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --base_path ${base_path} \
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
