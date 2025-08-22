from functools import cache
import torch
from torch import nn
import math
import einops
from torch import Tensor
from jaxtyping import Float, Int
import einx
import os
import json
class Linear(nn.Module):
    def __init__(self, input_size, output_size,bias = False):
        super(Linear, self).__init__()
        std = math.sqrt(2.0 / (input_size + output_size))
        weight = torch.randn(input_size, output_size, requires_grad=True)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weight = torch.nn.Parameter(weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(output_size, requires_grad=True))
        else:
            self.bias = None
        

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None}"

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = torch.randn(num_embeddings, embedding_dim,  requires_grad=True)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=3)
        self.weight = torch.nn.Parameter(weight)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / RMS).to(in_dtype)

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super(RoPE, self).__init__()
        self.register_buffer('freqs_cis',
                             self._init_cache(max_seq_len, d_k, theta),
                             persistent=False)
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float):
        d = torch.arange(0, dim, 2) / dim
        freqs = torch.exp(-d * math.log(theta))
        t = torch.arange(0, context_length)
        freqs = torch.einsum("i, j -> i j", t, freqs)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, "... seq d"], token_positions: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq d"]:
        x1, x2 = einx.rearrange("... (half_d xy) -> xy ... half_d",x, xy=2) # ... seq half_d
        
        cos, sin = self.freqs_cis[0, token_positions, :], self.freqs_cis[1, token_positions, :] 
        # Add num_heads dimension: [batch_size, seq_len, d_k//2] -> [batch_size, 1, seq_len, d_k//2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        res :torch.Tensor = einx.rearrange("... half_d, ... half_d -> ... (half_d (1+1))", x1_rot, x2_rot)
        return res.contiguous()
        
        

def Softmax(x: torch.Tensor, dim = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x_shifted = torch.exp(x - x_max)
    return exp_x_shifted / torch.sum(exp_x_shifted, dim=dim, keepdim=True)

def ScaledDotProductAttention(Q, K, V, mask = None, dropout = None):
    d_k = K.size(-1)
    attention_matrix = torch.einsum("... q d, ... k d -> ... q k", Q, K) / math.sqrt(d_k)
    if mask is not None:
        attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))
    attention_weights = Softmax(attention_matrix, dim = -1)
    if dropout is not None:
        attention_weights = nn.functional.dropout(attention_weights, dropout)
    output = torch.einsum("... q k, ... k d -> ... q d", attention_weights, V)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta = None, max_seq_len = None, dropout = None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_proj = Linear(d_model, d_model)
        self.q_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.dropout = dropout
        self.rope = RoPE(self.d_k, theta, max_seq_len) if theta is not None and max_seq_len is not None else lambda x, _: x
    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q, K, V = (einx.rearrange("... seq (h d) -> ... h seq d", X, h=self.num_heads) for X in (Q, K, V))
        
        if token_positions is None:
            token_positions = einx.rearrange("... seq -> b ... seq", torch.arange(seq_len), b = batch_size)
        # token_positions = einx.rearrange("... seq ->1 ... seq ", token_positions)
        # 线性变换并分头
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)
        # 计算注意力
        output = ScaledDotProductAttention(Q, K, V, mask, dropout=self.dropout) # (b ,h ,q, d))
        
        # 合并多头
        output = einx.rearrange("... h q d -> ... q (h d)", output).contiguous()
        
        # 最后的线性变换
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, attn_dropout: float | None = None, ffn_dropout:float | None = None, theta = None, max_seq_len = None):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout1 = nn.Dropout(attn_dropout) if attn_dropout is not None else lambda x: x
        self.dropout2 = nn.Dropout(ffn_dropout) if ffn_dropout is not None else lambda x: x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.dropout1(self.attn(self.ln1(x), token_positions))
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,context_length:int, num_heads: int, num_layers: int,  d_model: int, d_ff: int,attn_dropout: float | None= None, ffn_dropout: float = None, theta = None):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, attn_dropout, ffn_dropout, theta=theta, max_seq_len=context_length) for _ in range(num_layers)
            ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) 
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 1.0, top_k: int | None = None, eos_token_id = None) -> torch.LongTensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        self.eval()
        x = input_ids.clone()
        origin_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            logits = self.forward(x_cond) 
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = Softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            x = torch.cat([x, next_token], dim=-1)
        return torch.LongTensor(x[:, origin_sequence_length:])
    @classmethod
    def from_pretrained(cls, model_id: str):
        config_path = os.path.join(model_id, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(os.path.join(model_id, 'model.pt'), weights_only=True)
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
    
    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        model_state_path = os.path.join(save_path, 'model.pt')
        torch.save(self.state_dict(), model_state_path)
        config_path = os.path.join(save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        