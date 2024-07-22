from logging import warning as warn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from utils import (
    setup_logger,
    arg_parse
)
from SFTrainer import (
    SFTrainer
)
from dataload import make_KC_data_module
from peft import LoraConfig, get_peft_model

'''
llama 다운받고 왜 재사용안돼지? 왜 지랄?
inference code 뵈야함, prompt template 바꿔야함

데이터 강제 filtering 함 24GB 2개 부족

Hyper-Parameter Tuning -> Grid Search
    - baseline LR, LoRA Setting

tokenizer padding side 찍어보기

model name -> vicuna, llama2-chat
'''

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
        
        
if __name__ == '__main__':
    args = arg_parse()
    # setup_logger(args.output_dir)
    
    train_batch = 4
    eval_batch = 1
    accumulation = 2
    lora_alpha = args.lora_alpha
    weight_decay = args.weight_decay

    model_name = args.model_name
    output_dir = args.output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    '''
    data_module
        train_dataset
        eval_dataset
        data_collator
    '''
    data_module = make_KC_data_module(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        # target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        target_modules=['q_proj', 'k_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    trainingArgs = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = train_batch, 
        per_device_eval_batch_size = eval_batch,
        gradient_accumulation_steps = accumulation,
        gradient_checkpointing=True,
        
        optim = "adamw_hf",
        save_steps = 400,
        eval_steps = 400,
        logging_steps = 1,
        max_grad_norm = 0.7,  # for gradient clipping
        num_train_epochs=1,
        evaluation_strategy="steps", # epoch? or steps?
        save_strategy='steps',
        learning_rate=args.lr,
        weight_decay=weight_decay,
        dataloader_num_workers=16,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        remove_unused_columns=False
    )
    # warn(trainingArgs)
    
    trainer = SFTrainer(
        model=model, 
        args=trainingArgs,
        tokenizer=tokenizer,
        **data_module,
    )
    trainer.train()