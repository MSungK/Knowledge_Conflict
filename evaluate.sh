base_path="custom_training/grid_search"  # Shoud specify the path in python scirpt(infer.py)
check=165

for adapter_path in SFT_max_grad:0.7_wd:0.01_alpha16  SFT_max_grad:0.7_wd:0.1_alpha64   SFT_max_grad:1.3_wd:0.1_alpha16   SFT_max_grad:1.7_wd:0.01_alpha64SFT_max_grad:0.7_wd:0.01_alpha32  SFT_max_grad:0.7_wd:0.1_alpha8    SFT_max_grad:1.3_wd:0.1_alpha32   SFT_max_grad:1.7_wd:0.01_alpha8SFT_max_grad:0.7_wd:0.01_alpha64  SFT_max_grad:1.3_wd:0.01_alpha16  SFT_max_grad:1.3_wd:0.1_alpha64   SFT_max_grad:1.7_wd:0.1_alpha16SFT_max_grad:0.7_wd:0.01_alpha8   SFT_max_grad:1.3_wd:0.01_alpha32  SFT_max_grad:1.3_wd:0.1_alpha8    SFT_max_grad:1.7_wd:0.1_alpha32SFT_max_grad:0.7_wd:0.1_alpha16   SFT_max_grad:1.3_wd:0.01_alpha64  SFT_max_grad:1.7_wd:0.01_alpha16  SFT_max_grad:1.7_wd:0.1_alpha64SFT_max_grad:0.7_wd:0.1_alpha32   SFT_max_grad:1.3_wd:0.01_alpha8   SFT_max_grad:1.7_wd:0.01_alpha32  SFT_max_grad:1.7_wd:0.1_alpha8
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