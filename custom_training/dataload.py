from logging import warning as warn
from torch.utils.data import Dataset
from datasets import (
    load_dataset, 
    Value, 
    DatasetDict, 
    Features,
)
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Dict, Sequence
from functools import partial
import os
from copy import deepcopy
import torch
from os import path as osp
IGNORE_INDEX = -100

def generate_qa_instruction_prompt_with_ct(question, document):
    prompt = f'Instruction: answer the question based on the given context.\nQ: {question}?\nContext: {document}\nA: '
    return prompt


def generate_qa_instruction_prompt_without_ct(question):
    prompt = f'Instruction: answer the question based on the given context.\nQ: {question}?\nContext: None\nA: '
    return prompt


def _tokenize_fn(text: str, tokenizer: AutoTokenizer, max_seq_length: int) -> Dict:
    input_ids = labels = tokenizer(
        text,
        return_tensors='pt',
        padding='longest',
        max_length=max_seq_length,
        truncation=True,
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def prompt_fn(example, tokenizer, max_seq_length):   
    
    question = example['question']
    substituted_context = example['substituted_context']
    substituted_answer = example['substituted_answers']
    assert len(substituted_answer) == 1
    substituted_answer = substituted_answer[0]
    
    w_ct_prompt = generate_qa_instruction_prompt_with_ct(question, substituted_context)
    wt_ct_prompt = generate_qa_instruction_prompt_without_ct(question)
    answer = substituted_answer + tokenizer.eos_token
    
    w_ct_prompt_a = _tokenize_fn(w_ct_prompt + answer, tokenizer=tokenizer, max_seq_length=max_seq_length)
    w_ct_prompt = _tokenize_fn(w_ct_prompt, tokenizer=tokenizer, max_seq_length=max_seq_length)
    wt_ct_prompt_a = _tokenize_fn(wt_ct_prompt + answer, tokenizer=tokenizer, max_seq_length=max_seq_length)
    wt_ct_prompt = _tokenize_fn(wt_ct_prompt, tokenizer=tokenizer, max_seq_length=max_seq_length)
    
    w_ct_input_ids = w_ct_prompt_a['input_ids'].flatten()
    w_ct_len = w_ct_prompt['input_ids_lens']
    w_ct_labels = deepcopy(w_ct_input_ids)
    w_ct_labels[:w_ct_len-1] = IGNORE_INDEX
    w_ct_attention_mask = torch.ones_like(w_ct_input_ids)
    
    wt_ct_input_ids = wt_ct_prompt_a['input_ids'].flatten()
    wt_ct_len = wt_ct_prompt['input_ids_lens']
    wt_ct_labels = deepcopy(wt_ct_input_ids)
    wt_ct_labels[:wt_ct_len-1] = IGNORE_INDEX
    wt_ct_attention_mask = torch.ones_like(wt_ct_input_ids)
    
    w_ct = dict(
        input_ids=w_ct_input_ids,
        labels=w_ct_labels,
        attention_mask=w_ct_attention_mask
    )
    wt_ct = dict(
        input_ids=wt_ct_input_ids,
        labels=wt_ct_labels,
        attention_mask=wt_ct_attention_mask
    )
    return {
        'w_ct' : w_ct,
        'wt_ct' : wt_ct
    }
    
    
class KCDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, filtered: int):
        super(KCDataset, self).__init__()
        warn('Loading Data')
        self.dataset = load_dataset("json", data_files=data_path)['train']
        self.dataset = self.dataset.map(
            function=partial(prompt_fn, tokenizer=tokenizer, max_seq_length=4096), # TODO max_seq_length control
            batched=False,
            load_from_cache_file=False,
            drop_last_batch=False,
            num_proc=os.cpu_count() // 2,
            remove_columns=self.dataset.features.keys(),
            desc="Reformatting and Tokenizing")
        self.dataset.set_format(type='pt')
        print(self.dataset)
        

        # # TODO
        # len_list = list()
        # for i in self.dataset:
        #     dat = i['w_ct']['input_ids'].shape[0]
        #     len_list.append(dat)
        
        # file_name = data_path.split('/')[-1]
        # torch.save(len_list, f'{file_name}.pt')
        # print(f'Saved in {file_name}.pt')
        
        self.dataset = self.dataset.filter(
            lambda example: example['w_ct']['input_ids'].shape[0] < filtered
        )
        # self.dataset = self.dataset.filter(
        #     lambda example: example['w_ct']['input_ids'].shape[0] > 1000
        # )
        print('Filtered')
        print(self.dataset)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

@dataclass
class DataCollatorForKCDataset(object):
    tokenizer: AutoTokenizer
    def __call__(self, instances):
        w_ct_input_ids, w_ct_labels, w_ct_attention_masks = tuple([instance['w_ct'][key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        wt_ct_input_ids, wt_ct_labels, wt_ct_attention_masks = tuple([instance['wt_ct'][key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        
        w_ct_input_ids = torch.nn.utils.rnn.pad_sequence(
            w_ct_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        w_ct_labels = torch.nn.utils.rnn.pad_sequence(
            w_ct_labels, batch_first=True, padding_value=-100
        )
        w_ct = dict(
            input_ids = w_ct_input_ids,
            labels = w_ct_labels,
            attention_mask = w_ct_input_ids.ne(self.tokenizer.pad_token_id)
        )
        
        wt_ct_input_ids = torch.nn.utils.rnn.pad_sequence(
            wt_ct_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        wt_ct_labels = torch.nn.utils.rnn.pad_sequence(
            wt_ct_labels, batch_first=True, padding_value=-100
        )
        wt_ct = dict(
            input_ids = wt_ct_input_ids,
            labels = wt_ct_labels,
            attention_mask = wt_ct_input_ids.ne(self.tokenizer.pad_token_id)
        )
        
        return dict(
            w_ct=w_ct,
            wt_ct=wt_ct
        )
        

def make_KC_data_module(tokenizer: AutoTokenizer) -> Dict:
    eval_data_path = '../datasets/data/MRQANaturalQuestionsDev-closedbookfiltered-corpus-counterfactual.json'
    eval_dataset = KCDataset(tokenizer=tokenizer, data_path = eval_data_path, filtered=6000)    
    
    train_data_path = '../datasets/data/MRQANaturalQuestionsTrain-closedbookfiltered-corpus-counterfactual.json'
    train_dataset = KCDataset(tokenizer=tokenizer, data_path = train_data_path, filtered=1200)    
    
    data_collator = DataCollatorForKCDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)