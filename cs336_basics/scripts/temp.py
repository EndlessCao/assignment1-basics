from cs336_basics.tokenizer import Tokenizer
import numpy as np
import os
import gc
from tqdm import tqdm

dataset = {
    'name': 'tinystories',
    'train': '/home/caowei/workspace/assignment1-basics/TinyStoriesV2-GPT4-train.txt',
    'valid': '/home/caowei/workspace/assignment1-basics/TinyStoriesV2-GPT4-valid.txt',
}
vocab = {
    "vocab_path": "/home/caowei/workspace/assignment1-basics/tests/fixtures/gpt2_vocab.json",
    "merges_path": "/home/caowei/workspace/assignment1-basics/tests/fixtures/gpt2_merges.txt",
    "special_tokens": [""],
}

# 确保数据目录存在
data_dir = '/home/caowei/workspace/assignment1-basics/cs336_basics/data'
os.makedirs(data_dir, exist_ok=True)

tokenizer = Tokenizer.from_files(**vocab)



