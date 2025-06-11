from dataclasses import dataclass, field
import torch
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import math
import os
from typing import BinaryIO, IO
from jaxtyping import Float, Int
from data import DataLoader
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def CrossEntropyLoss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    # inputs: prob_logits
    batch_size = inputs.shape[0]
    
    # 对输入进行log_softmax处理
    max_logits = torch.max(inputs, dim=1, keepdim=True).values
    # torch.max(inputs, dim=1, keepdim=True)返回一个元组，第一个元素是最大值，第二个元素是最大值的索引
    inputs = inputs - max_logits
    log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
    
    # 获取目标类别对应的概率
    targets = targets.view(-1)
    loss = -log_probs[range(batch_size), targets].mean()
    
    return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), weight_decay = 0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
            
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  # 一阶动量
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # 二阶动量
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                
                # 更新一阶动量和二阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) # m = beta1 * m + (1 - beta1) * grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v = beta2 * v + (1 - beta2) * grad * grad
                
                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state["step"] # (1 - beta1 ** t)
                bias_correction2 = 1 - beta2 ** state["step"] # (1 - beta2 ** t)
                
                # 更新参数
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1 # lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                denom = exp_avg_sq.sqrt().add_(eps)  # sqrt(v) + eps
                p.data.addcdiv_(exp_avg, denom, value=-step_size) # p = p - step_size * m / (sqrt(v) + eps ) / sqrt(1 - beta2 ** t)
                
                # AdamW的权重衰减
                p.data.add_(p.data, alpha=-lr * weight_decay)
                
                
        return loss
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    
class CosineScheduler:
    def __init__(self, amax, amin, Tw, Tc):
        self.amax = amax
        self.amin = amin
        self.Tw = Tw
        self.Tc = Tc
    def __call__(self, t):
        if t < self.Tw:
            return self.amax * t / self.Tw
        if t <= self.Tc:
            return self.amin + 0.5 * (1 + math.cos((t - self.Tw) / (self.Tc - self.Tw) * math.pi)) * (self.amax - self.amin)
        return self.amin


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    # 计算所有参数的梯度的L2范数
    total_norm = torch.norm(torch.stack([p.grad.detach() for p in parameters if p.grad is not None]), 2)
    
    # 如果总范数大于最大允许范数,则进行截断
    clip_coef = max_l2_norm / (total_norm + eps)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    
    # 对所有参数的梯度进行截断
    for p in parameters:
        if p.grad is not None:
            p.grad.detach().mul_(clip_coef)
            
def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]):
    obj = {}
    obj['model'] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    obj["iteration"] = iteration
    torch.save(obj, out)
    pass

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    iteration = obj['iteration']
    return iteration


    
    

class Trainer:
    def __init__(self, model, optimizer, scheduler, tokenizer, config, data_loader: DataLoader):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = config.batch_size
        self.context_length = config.context_length
        self.device = config.device
        self.train_config = config
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.total_iters = config.total_iters
        self.log_interval = config.log_interval
        self.eval_interval = config.eval_interval
        self.eval_iters = config.eval_iters
        self.iter_num = 0
    
    def train(self, ckpt=None):
        if ckpt is not None:
            iteration = load_checkpoint(ckpt, self.model, self.optimizer)
        else:
            iteration = 0
        self.iter_num = iteration
        for iteration in range(iteration, self.total_iters):
            x, y = self.data_loader.get_batch('train')
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = CrossEntropyLoss(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            gradient_clipping(self.model.parameters(), 1.0)
            lr = self.scheduler(iteration)
            self.optimizer.set_lr(lr)
            self.optimizer.step()
            
            if iteration % self.log_interval == 0:
                logging.info(f"iteration {iteration}: loss {loss.item()} lr {lr}")
            if iteration % self.eval_interval == 0:
                self.eval()
                save_checkpoint(self.model, self.optimizer, iteration, f"checkpoint_{iteration}.pt")
            self.iter_num += 1
    def eval(self):
        total_loss = 0
        for _ in range(self.eval_iters):
            x, y = self.data_loader.get_batch('val')
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                loss = CrossEntropyLoss(logits, y)
                total_loss += loss.item()
        avg_loss = total_loss / self.eval_iters
        logging.info(f"Iter {self.iter_num}: eval loss {avg_loss}")
        return avg_loss
        