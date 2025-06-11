from tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
dataset = {
    'name': 'tinystories',
    'train': '/Users/caowei/Workspace/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt',
    'valid': '/Users/caowei/Workspace/assignment1-basics/tests/fixtures/tinystories_sample.txt',
}
vocab = {
    "vocab_path": "/Users/caowei/Workspace/assignment1-basics/tests/fixtures/tiny_vocab.json",
    "merges_path": "/Users/caowei/Workspace/assignment1-basics/tests/fixtures/tiny_merges.txt",
    "special_tokens": ["<|endoftext|>"],
}

tokenizer = Tokenizer.from_files(**vocab)

for split in ['train', 'valid']:
    with open(dataset[split]) as f:
        text = f.read()
    
    encoded = tokenizer.encode(text, progress_bar=True)
    
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(f"/Users/caowei/Workspace/assignment1-basics/cs336_basics/data/{dataset['name']}_{split}.bin", dtype=np.uint16, mode='w+', shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc = f'writing {split}'):
        batch = encoded[idx:idx+batch_size]
        arr[idx:idx+batch_size] = batch
        idx += batch_size
arr.flush