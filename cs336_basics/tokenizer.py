from dataclasses import dataclass
import os
from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN, get_tokenizer_from_vocab_merges_path
import regex as re
from typing import Dict, Sequence, Tuple, Iterable, List
from dataclasses import dataclass
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import chain


def update(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and tuple(ids[i : i + 2]) == pair:
            new_ids.append(new_id)
            i += 2  # 应该增加2，因为我们已经处理了一对token
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def get_pairs(ids: Sequence[int]):
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs


@dataclass
class _Vocab:
    token_to_id: Dict[bytes, int]
    id_to_token: Dict[int, bytes]


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: Iterable[Tuple[bytes, bytes]],
        special_tokens: Iterable[str] | None = None,
    ):
        id_to_token = vocab.copy()
        token_to_id = {bytes_token: token_id for token_id, bytes_token in vocab.items()}
        self.vocab = _Vocab(token_to_id, id_to_token)

        for i in range(256):
            byte = bytes([i])
            if byte not in self.vocab.token_to_id:
                self.vocab.token_to_id[byte] = len(self.vocab.token_to_id)
                self.vocab.id_to_token[len(self.vocab.id_to_token)] = byte

        self.merges = {}
        for a, b in merges:
            merged = (self.vocab.token_to_id[a], self.vocab.token_to_id[b])
            self.merges[merged] = self.vocab.token_to_id[a + b]

        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab.token_to_id:
                    self.vocab.token_to_id[token_byte] = len(self.vocab.token_to_id)
                    self.vocab.id_to_token[len(self.vocab.id_to_token)] = token_byte
                    self.special_tokens[token] = len(self.vocab.id_to_token)
                else:
                    self.special_tokens[token] = self.vocab.token_to_id[token_byte]

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: Iterable[str] | None = None,
        **kwargs,
    ):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    def encode(
        self, text: str, return_tensors=False, progress_bar=False, num_workers: int = 1
    ) -> list[int] | torch.Tensor:
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens.keys()) + ")"
            chunks = re.split(pattern, text)
            chunks = [chunk for chunk in chunks if chunk]
        else:
            chunks = [text]
        input_ids = []
        with ThreadPoolExecutor(max_workers=min(max(len(chunks) // 4, 1), num_workers)) as executor:
            futures = [executor.submit(self._tokenize, chunk) for chunk in chunks]
            for future in tqdm(futures, desc="adding", disable=not progress_bar):
                input_ids.extend(future.result())
        if return_tensors:
            return torch.tensor(input_ids)
        return input_ids

    def encode_iterable(self, texts: Iterable[str]):
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
            ids = [self.vocab.token_to_id[bytes([b])] for b in chunk.encode("utf-8")]
            while len(ids) >= 2:
                pairs = get_pairs(ids)
                high_priority_pair = min(
                    pairs, key=lambda pair: self.merges.get(pair, float("inf"))
                )
                if high_priority_pair not in self.merges:
                    break
                new_id = self.merges[high_priority_pair]
                ids = update(ids, high_priority_pair, new_id)
            result.extend(ids)
        return result

    def decode(self, ids: List[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # 将字节序列连接并解码为字符串
        return b"".join(self.vocab.id_to_token[i] for i in ids).decode(
            "utf-8", errors="replace"
        )

    @property
    def vocab_size(self):
        return len(self.vocab.id_to_token)

    @property
    def vocab_id2token(self) -> dict:
        return self.vocab.id_to_token
