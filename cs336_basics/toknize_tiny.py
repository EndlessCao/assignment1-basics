from typing import Iterable
import numpy as np
import time
from tqdm import tqdm
import pathlib
import time
from tokenizer import Tokenizer
DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'
tinystory = {
    'train':DATA_PATH / 'TinyStoriesV2-GPT4-train.txt',
    'val':DATA_PATH / 'TinyStoriesV2-GPT4-valid.txt',
    'vocab_path': DATA_PATH / 'tiny_vocab.json',
    'merges_path': DATA_PATH / 'tiny_merges.txt',
    'special_tokens': ['<|endoftext|>']
}


tokenizer = Tokenizer.from_files(**tinystory)

for split in ['train', 'val']:
    with open(tinystory[split]) as f:
        text = f.read()
    print(f'Encoding {split} set')
    start_time = time.time()
    encoded = tokenizer.encode(text, progress_bar=True, num_workers=1)
    end_time = time.time()
    print(f'Encoding {split} set takes {end_time - start_time} seconds')
    # save the ids
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(DATA_PATH / f'ts_{split}.bin', dtype=np.uint16, mode='w+', shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
        batch = encoded[idx:idx+batch_size]
        arr[idx:idx+batch_size] = batch
        idx += batch_size
    arr.flush()
