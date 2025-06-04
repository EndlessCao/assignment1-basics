from dataclasses import dataclass
from itertools import pairwise
from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN
from cs336_basics.utils.io import get_tokenizer_from_vocab_merges_path
import regex as re
from typing import Dict, Tuple, Iterable, List
@dataclass
class _Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: Iterable[Tuple[bytes, bytes]], special_tokens: Iterable[str]=None):
        id_to_token = vocab
        token_to_id = {token: i for i, token in vocab.items()}
        self.vocab = _Vocab(token_to_id, id_to_token)
        
        for i in range(256):
            byte = bytes([i])
            if byte not in self.vocab.token_to_id:
                self.vocab.token_to_id[byte] = len(self.vocab.token_to_id)
                self.vocab.id_to_token[len(self.vocab.id_to_token)] = byte
        
        self.merges = {}
        for a, b in merges:
            merged = (self.vocab.token_to_id[a] , self.vocab.token_to_id[b])
            self.merges[merged] = self.vocab.token_to_id[a+b]
        
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode('utf-8')
                if token_byte not in self.vocab.token_to_id:
                    self.vocab.token_to_id[token_byte] = len(self.vocab.token_to_id)
                    self.vocab.id_to_token[len(self.vocab.id_to_token)] = token_byte
                    self.special_tokens[token] = self.vocab.token_to_id[token_byte]
                else:
                    self.special_tokens[token] = self.vocab.token_to_id[token_byte]
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens:Iterable[str]=None):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = '(' + '|'.join(re.escape(token) for token in self.special_tokens.keys()) + ')'
            chunks = re.split(pattern, text)
            chunks = [chunk for chunk in chunks if chunk]
        else:
            chunks = [text]
        input_ids = []
        for chunk in chunks:
            input_ids += self._tokenize(chunk)
        return input_ids
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterable[list[int]]:
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id
            
    def _tokenize(self, text: str) -> list[int]:
        # 如果是特殊token直接返回对应id
        if text in self.special_tokens:
            return [self.special_tokens[text]]
            
        # 使用GPT2预分词模式切分文本
        text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        result = []
        
        # 处理每个文本块
        for chunk in text_chunks:
            # 将文本块转换为UTF-8字节序列,并映射到对应的token id
            ids = [self.vocab.token_to_id[bytes([b])] for b in chunk.encode('utf-8')]
            
            # 当序列长度大于1时,尝试应用BPE合并规则
            while len(ids) > 1:
                # 获取所有相邻token对
                pairs = set(pairwise(ids))
                # 找到优先级最高的合并对(merge分数最小)
                best_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
                
                # 如果没有可用的合并规则则退出
                if best_pair not in self.merges:
                    break
                    
                best_pair_id = self.merges[best_pair]
                
                # 内部函数:执行token对的合并操作
                def update(pair: Tuple[int, int], new_id: int):
                    nonlocal ids
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        curr_pair = tuple(ids[i:i+2])
                        if curr_pair == pair:
                            new_ids.append(new_id)
                            i += 1
                        else:
                            new_ids.append(ids[i])
                        i += 1
                    ids = new_ids
                
                # 执行最佳合并对的合并
                update(best_pair, best_pair_id)
            
            # 将处理后的id添加到结果中
            result += ids
            
        return result
    def decode(self, ids: List[int]) -> str:
        # 将字节序列连接并解码为字符串
        return b''.join(self.vocab.id_to_token[i] for i in ids).decode('utf-8', errors='replace')