import torch

from .nn import TransformerLM
from .trainer import *
from .data import *
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class TrainConfig:
    # dataset
    dataset: str
    batch_size: int
    device: str
    
    # model
    vocab_size: Optional[int] = field(default=50257)
    context_length: Optional[int] = field(default=1024)
    num_layers: Optional[int] = field(default=12)
    d_model: Optional[int] = field(default=768)
    num_heads: Optional[int] = field(default=12)
    d_ff: Optional[int] = field(default=3072)
    attn_dropout: Optional[float] = field(default=0.1)
    ffn_dropout: Optional[float] = field(default=0.1)
    theta: Optional[int] = field(default=10000)
    # training
    total_iters: Optional[int] = field(default=10*(10**3))
    warmup_iters: Optional[int] = field(default=None)
    lr_max: Optional[float] = field(default=5e-4)
    lr_min: Optional[float] = field(default=0)
    weight_decay: Optional[float] = field(default=0.001)
    
    # logging parameters
    wandb_logging: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_interval: Optional[int] = field(default=None)
    eval_interval: Optional[int] = field(default=None)
    eval_iters: Optional[int] = field(default=100)

train_config = TrainConfig(
    
)
train_data = Dataset("data/train.bin")
val_data = Dataset("data/val.bin")

model = TransformerLM(
    **asdict(TrainConfig)
)





