python infer.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --data_path datasets/data/MRQANaturalQuestionsSPLIT-closedbookfiltered-corpus-counterfactual.json \
    --output_file ./NaturalQuestions_output.json \
    --device auto \
    --num_shards 1 \
    --shard_id 0 \
    --add_eot_token \
    --prompt_template "instruction-based"