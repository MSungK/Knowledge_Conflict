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

    return parser.parse_args()