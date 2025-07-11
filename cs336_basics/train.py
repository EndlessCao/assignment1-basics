import torch

from nn import TransformerLM
from trainer import *
from cs336_basics.utils.data import DataLoader,Dataset

from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class TrainConfig:
    # dataset
    dataset: str = "tinystories"
    batch_size: int = 16 # 进一步减小批处理大小，从4减小到2
    device: str = "cuda"
    context_length: Optional[int] = field(default=1024)
    # training
    total_iters: Optional[int] = field(default=10*(10**3))
    warmup_iters: Optional[int] = field(default=10)
    lr_max: Optional[float] = field(default=5e-4)
    lr_min: Optional[float] = field(default=0)
    weight_decay: Optional[float] = field(default=0.001)
    
    # logging parameters
    wandb_logging: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_interval: Optional[int] = field(default=10)
    eval_interval: Optional[int] = field(default=100)
    eval_iters: Optional[int] = field(default=100)
@dataclass
class ModelConfig:
    vocab_size: Optional[int] = field(default=50257)
    context_length: Optional[int] = field(default=256)
    num_layers: Optional[int] = field(default=6)
    d_model: Optional[int] = field(default=384)
    num_heads: Optional[int] = field(default=6)
    d_ff: Optional[int] = field(default=1024)
    attn_dropout: Optional[float] = field(default=0.1)
    ffn_dropout: Optional[float] = field(default=0.1)
    theta: Optional[int] = field(default=10000)
    

train_config = TrainConfig()
model_config = ModelConfig()
train_data = Dataset("/home/caowei/workspace/assignment1-basics/data/train.npy")
val_data = Dataset("/home/caowei/workspace/assignment1-basics/data/valid.npy")

data_loader = DataLoader(train_data, val_data, batch_size=train_config.batch_size, context_length=model_config.context_length, device=train_config.device)
model = TransformerLM(
    **asdict(model_config),device=train_config.device
)
print(model)
print(model.device)
# 确保模型在CUDA设备上
model = model.to(train_config.device)
print(f"模型现在在设备: {next(model.parameters()).device}")
optimizer = AdamW(model.parameters())
scheduler = CosineScheduler(train_config.lr_max, train_config.lr_min, train_config.warmup_iters, train_config.total_iters)
trainer = Trainer(model, optimizer=optimizer,scheduler=scheduler, config=train_config, data_loader=data_loader)

trainer.train()




