import torch
from torch import nn
import math
from einops import rearrange, einsum
import einx
from torch import Tensor
from jaxtyping import Float, Bool, Int
class Linear(nn.Module):
    def __init__(self, input_size, output_size, device = 'cpu', dtype = torch.float16, bias = False):
        super(Linear, self).__init__()
        # 使用Xavier初始化，更适合深度网络
        self.weight = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.zeros(input_size, output_size, device=device, dtype=dtype))
        )
    def forward(self, x):
        out = torch.matmul(x, self.weight)
        return out

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device='cpu', dtype=torch.float32):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 使用正态分布初始化，但缩小标准差以适应float16
        self.weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype), mean=0.0, std=0.02)
        )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device='cpu', dtype=torch.float32):
        super(RMSNorm, self).__init__() # 添加了这一行
        self.device = device
        self.d_model = d_model
        self.eps = eps
        # 使用ones初始化，这是RMSNorm的标准做法
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        RMS = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / RMS)

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device='cpu', dtype=torch.float32):
        super(SwiGLU, self).__init__()
        self.device = device
        self.d_model = d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None): # 参数顺序已调整
        super(RoPE, self).__init__()
        self.d_k = d_k  # 保存 d_k 为实例属性
        self.theta = theta  # 保存 theta 为实例属性
        self.max_seq_len = max_seq_len
        self.device = device

        # 生成位置编码的m矩阵
        m = torch.arange(0, self.max_seq_len, device=self.device).unsqueeze(1) # (max_seq_len, 1)
        # 生成位置编码的theta矩阵
        # 使用 self.d_k 和 self.theta
        theta_matrix = torch.exp(
            torch.arange(0, self.d_k, 2, device=self.device).float() * (-math.log(self.theta) / self.d_k)
        )
        
        # 计算旋转位置编码
        # 使用 self.d_k
        self.freqs_cis = torch.zeros((self.max_seq_len, self.d_k), device=self.device)
        for i in range(0, self.d_k, 2): # 使用 self.d_k
            self.freqs_cis[:, i] = torch.cos(m * theta_matrix[i//2]).squeeze(-1)
            self.freqs_cis[:, i+1] = torch.sin(m * theta_matrix[i//2]).squeeze(-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 获取当前序列长度的位置编码
        freqs = self.freqs_cis[token_positions]
        
        # 将位置编码应用到输入张量
        x_rotated = torch.zeros_like(x)
        
        # 处理多头注意力的情况，token_positions形状为[batch_size, seq_len]，而x形状为[batch_size, num_heads, seq_len, d_k]
        if x.dim() == 4 and token_positions.dim() == 2:
            # 扩展freqs以匹配x的维度：[batch_size, seq_len, d_k] -> [batch_size, 1, seq_len, d_k]
            freqs = freqs.unsqueeze(1)
            
        for i in range(0, self.d_k, 2):
            x_rotated[..., i] = x[..., i] * freqs[..., i] - x[..., i+1] * freqs[..., i+1]
            x_rotated[..., i+1] = x[..., i+1] * freqs[..., i] + x[..., i] * freqs[..., i+1]
        
        return x_rotated

class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"
def Softmax(x: torch.Tensor, dim = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x_shifted = torch.exp(x - x_max)
    return exp_x_shifted / torch.sum(exp_x_shifted, dim=dim, keepdim=True)

def ScaledDotProductAttention(Q, K, V, mask = None, dropout = None):
    attention_matrix = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        attention_matrix = attention_matrix.masked_fill(mask == 0, -1e4)  # Changed from -1e9 to -1e4 to avoid overflow in float16
    attention_weights = Softmax(attention_matrix, dim = -1)
    if dropout is not None:
        attention_weights = nn.functional.dropout(attention_weights, dropout)
    output = torch.matmul(attention_weights, V)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=torch.float32, dropout = None, positional_embedding = None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.dropout = dropout
        self.pos_embedding = positional_embedding if positional_embedding is not None else lambda x, _: x
    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.pos_embedding(self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2), token_positions)
        k = self.pos_embedding(self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2), token_positions)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).unsqueeze(0).unsqueeze(0)
        output = ScaledDotProductAttention(q, k, v, mask, dropout=self.dropout)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 最后的线性变换
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, attn_dropout: float = None, ffn_dropout:float = None, device=None, dtype=torch.float32, max_seq_len = None, pos_embedding = None):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, device=device, dtype=dtype, max_seq_len=max_seq_len, positional_embedding=pos_embedding)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(attn_dropout) if attn_dropout is not None else lambda x: x
        self.dropout2 = nn.Dropout(ffn_dropout) if ffn_dropout is not None else lambda x: x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.dropout1(self.attn(self.ln1(x), token_positions))
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x

class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 context_length:int, 
                 num_heads: int, 
                 num_layers: int,  
                 d_model: int, 
                 d_ff: int,
                 attn_dropout: float = None, 
                 ffn_dropout: float = None, 
                 device=None, 
                 dtype=torch.float32, 
                 theta = None):
        super(TransformerLM, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        #self.positional_embedding = Embedding(context_length, d_model, device=device, dtype=dtype)
        self.positional_embedding = RoPE(d_model, theta, context_length, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, attn_dropout, ffn_dropout, device=device, dtype=dtype, max_seq_len=context_length, positional_embedding=self.positional_embedding) 
            for _ in range(num_layers)
            ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(x) 
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
    def get_num_params():
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, eos_token_id = None) -> torch.LongTensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        x = input_ids.clone()
        origin_sequence_length = x.shape[-1]
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.context_length:]
            logits = self.forward(x_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = Softmax(logits)
            next_token = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            x = torch.cat([x, next_token], dim=-1)
        return x[:, origin_sequence_length:]

        