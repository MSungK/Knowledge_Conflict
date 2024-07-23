import logging


def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f'logs/{log_path}')])
    
    
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lora_alpha', type=int, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--max_grad_norm', type=float, required=True)
    parser.add_argument('--grid_search', action='store_true')

    return parser.parse_args()

import torch
import numpy as np
import random

def fix_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)