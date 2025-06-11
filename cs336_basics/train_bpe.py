from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import Counter, defaultdict
from itertools import pairwise
import regex as re
from typing import Iterable
import os

import multiprocessing as mp


def _find_pretokens(text: str):
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))


def read_text_file(input_path: str, num_worker: int, special_tokens: Iterable[str]):
    # Read the input text file
    with open(input_path, "r") as file:
        text = file.read()

    text_chunks = re.split("|".join(re.escape(token) for token in special_tokens), text)
    text_chunks = [chunk for chunk in text_chunks if chunk]

    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        pretokens = executor.map(_find_pretokens, text_chunks)
    pretokens = sum(pretokens, Counter())
    # 将每个pretoken转换为bytes元组的字典推导式
    pretoken_freq = {
        tuple(bytes([b]) for b in pretoken.encode("utf-8")): freq
        for pretoken, freq in pretokens.items()
    }

    return pretoken_freq


def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc : merge_loc + 2]
    suffix = byte_tuple[merge_loc + 2 :]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: Iterable[str],
    num_workers: int = mp.cpu_count() - 1,
):
    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {
        **{i: token.encode("utf-8") for i, token in enumerate(special_tokens)},
        **{i + len(special_tokens): bytes([i]) for i in range(256)},
    }

    # 预处理得到pretoken的频率
    pretoken_freq = read_text_file(input_path, num_workers, special_tokens)

    pair_freq = defaultdict(lambda: 0)
    for pretoken, freq in pretoken_freq.items():
        for i, j in pairwise(pretoken):
            pair_freq[(i, j)] += freq

    merges = []
    # train loop
    while len(vocab) < vocab_size:
        # Find the most frequent pair
        most_freq_pair = max(
            pair_freq, key=lambda k: (pair_freq[k], k)
        )  # 如果频率相同，选择字典序较大的pair

        # Add the pair to the merges list
        merges.append(most_freq_pair)

        # Update the vocab
        merged_pair = b"".join(most_freq_pair)
        vocab[len(vocab)] = merged_pair

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken, freq in pretoken_freq.items():
            for i in range(len(pretoken) - 1):
                pair = pretoken[i : i + 2]
                if pair == most_freq_pair:
                    pretoken, prefix, suffix = _update_byte_tuple(pretoken, i)
                    # Update the pair frequency table
                    if prefix:
                        add_pair = (prefix[-1], merged_pair)
                        pair_freq[add_pair] = pair_freq[add_pair] + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                    if suffix:
                        add_pair = (merged_pair, suffix[0])
                        pair_freq[add_pair] = pair_freq[add_pair] + freq
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                    pair_freq[most_freq_pair] -= freq
            # Update the pre-token frequency table
            new_pretoken_freq[pretoken] = freq
        pretoken_freq = new_pretoken_freq

    return vocab, merges


if __name__ == "__main__":
    # 简单测试
    from pathlib import Path
    import json
    import time

    FIXTURES_PATH = "/Users/caowei/Workspace/assignment1-basics/tests/fixtures"
    input_path = FIXTURES_PATH + "/" + "tinystories_sample_5M.txt"

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=1024,
        special_tokens=["<|endoftext|>"],  # 使用有意义的特殊标记
    )
    end_time = time.time()

    # 将词汇表写入文件
    vocab_path = FIXTURES_PATH + "/" + "tiny_vocab.json"

    # 将合并规则写入文件
    merges_path = FIXTURES_PATH + "/" + "tiny_merges.txt"

    from utils.io import save_voacb_and_merge
    save_voacb_and_merge(vocab, merges, vocab_path, merges_path)

    print(f"训练完成，用时: {end_time - start_time:.2f}秒")
    print(f"词汇表大小: {len(vocab)}")
    print(f"合并规则数量: {len(merges)}")
