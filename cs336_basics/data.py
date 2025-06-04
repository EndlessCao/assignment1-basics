from typing import Tuple
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import mmap
import os

def load_data(dataset, batch_size, context_length, device='cpu', num_worker = 1):

    # 计算可以采样的最大起始位置
    max_start = len(dataset) - context_length

    start_indices = np.random.randint(0, max_start, size=batch_size)
    
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        sequences = torch.stack(list(executor.map(lambda start: torch.LongTensor(dataset[start:start+context_length]).to(device), start_indices)))
        labels = torch.stack(list(executor.map(lambda start: torch.LongTensor(dataset[start+1:start+context_length+1]).to(device), start_indices)))
    
    return sequences, labels

class Dataset:
    def __init__(self, dataset_path):
        self.data = np.memmap(dataset_path, dtype=np.uint16, mode='r').astype(np.int64)
        self._transform_func = None  # 存储转换函数

    def __getitem__(self, index):
        item = self.data[index]
        if self._transform_func is not None:
            item = self._transform_func(item)  # 按需转换
        return item

    def __len__(self):
        return len(self.data)

    def map(self, func):
        self._transform_func = func  # 存储函数，不立即执行
        return self  # 返回自身，支持链式调用

class DataLoader:
    def __init__(self, train_data, val_data, batch_size, context_length, device='cpu', num_worker = 1):
        self.dataset = {"train": train_data, "val": val_data}
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.num_worker = num_worker
    
    def get_batch(self,split) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.dataset[split]
        return load_data(data, self.batch_size, self.context_length, self.device, self.num_worker)
        
        
        
