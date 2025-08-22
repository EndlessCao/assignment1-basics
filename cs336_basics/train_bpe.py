from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import regex as re
from typing import Tuple, Sequence
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


def _find_pretokens(text: str):
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))


def read_text_file(input_path: str, num_worker: int, special_tokens: Sequence[str]):
    # Read the input text file
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    text_chunks = re.split("|".join(re.escape(token) for token in special_tokens), text)
    text_chunks = [chunk for chunk in text_chunks if chunk]

    print(f"Processing {len(text_chunks)} text chunks with {num_worker} workers...")
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        pretokens = list(executor.map(_find_pretokens, text_chunks))
    

    print("Merging pretoken counters...")
    result = Counter()
    for counter in tqdm(pretokens, desc="Merging counters"):
        result.update(counter)
    pretokens = result
    
    print(f"Found {len(pretokens)} unique pretokens")
    
    # 将每个pretoken转换为bytes元组的字典推导式
    print("Converting pretokens to byte tuples...")
    pretoken_freq = {
        tuple(bytes([b]) for b in pretoken.encode("utf-8")): freq
        for pretoken, freq in tqdm(pretokens.items(), desc="Converting to bytes")
    }

    return pretoken_freq


def _update_byte_tuple(byte_tuple: Tuple[bytes, ...], merge_loc: int):
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
    special_tokens: Sequence[str],
    num_workers: int = mp.cpu_count() - 1,
):
    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {
        **{i: token.encode("utf-8") for i, token in enumerate(special_tokens)},
        **{i + len(special_tokens): bytes([i]) for i in range(256)},
    }

    # 预处理得到pretoken的频率
    print("Reading input file...")
    pretoken_freq = read_text_file(input_path, num_workers, special_tokens)
    
    # 初始化pair_freq
    pair_freq = defaultdict(int)

    for pretoken, freq in tqdm(pretoken_freq.items(), desc="Initializing pair frequency table"):
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_freq[pair] += freq
 

    merges = []
    # train loop
    initial_vocab_size = len(vocab)
    target_merges = vocab_size - initial_vocab_size
    print("Training BPE...")
    with tqdm(total=target_merges, desc="Training BPE") as pbar:
        while len(vocab) < vocab_size:
            # Find the most frequent pair
            most_freq_pair = max(
                pair_freq, key=lambda k: (pair_freq[k], k)
            )  # 如果频率相同，选择字典序较大的pair

            # Add the pair to the merges list
            merges.append(most_freq_pair)

            # Update the vocab
            new_id = max(vocab.keys()) + 1
            merged_pair = b"".join(most_freq_pair)
            vocab[new_id] = merged_pair
            # Update the pre-token frequency table and pair frequency table
            new_pretoken_freq = {}
            for pretoken, freq in pretoken_freq.items():
                for i in range(len(pretoken)):
                    pair = pretoken[i : i + 2]
                    if pair == most_freq_pair:
                        pretoken, prefix, suffix = _update_byte_tuple(pretoken, i)
                            # Update the pair frequency table
                        if prefix:
                            add_pair = (prefix[-1], merged_pair)
                            pair_freq[add_pair] += freq
                            del_pair = (prefix[-1], most_freq_pair[0])
                            pair_freq[del_pair] -= freq
                        if suffix:
                            add_pair = (merged_pair, suffix[0])
                            pair_freq[add_pair] += freq
                            del_pair = (most_freq_pair[1], suffix[0])
                            pair_freq[del_pair] -= freq
                        pair_freq[most_freq_pair] -= freq
                            
                    # Update the pre-token frequency table
                new_pretoken_freq[pretoken] = freq

            pretoken_freq = new_pretoken_freq
            # Update progress bar
            pbar.update(1)
    return vocab, merges


if __name__ == "__main__":
    # 简单测试
    from pathlib import Path
    import json
    import time
    import sys
    import datetime

    # 使用绝对路径
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_PATH = PROJECT_ROOT / "data"
    input_path = str(DATA_PATH / "TinyStoriesV2-GPT4-train.txt")

    # 设置参数
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_workers = 1

    print(f"Starting BPE training with parameters:")
    print(f"- Input file: {input_path}")
    print(f"- Target vocabulary size: {vocab_size}")
    print(f"- Special tokens: {special_tokens}")
    print(f"- Number of workers: {num_workers}")
    print("-" * 50)

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_workers=num_workers,
    )
    end_time = time.time()
    training_time = end_time - start_time
    
    # 将词汇表写入文件
    vocab_path = str(DATA_PATH / "tiny_vocab.json")
    merges_path = str(DATA_PATH / "tiny_merges.txt")

    print(f"Training completed in {training_time:.2f} seconds ({datetime.timedelta(seconds=int(training_time))})")
    print(f"Saving vocabulary to {vocab_path}")
    print(f"Saving merges to {merges_path}")

    from utils.io import save_voacb_and_merge
    save_voacb_and_merge(vocab, merges, vocab_path, merges_path)

    print("-" * 50)
    print(f"训练完成，用时: {training_time:.2f}秒")
    print(f"词汇表大小: {len(vocab)}")
    print(f"合并规则数量: {len(merges)}")
    print(f"文件已保存至: {DATA_PATH}")
    print("-" * 50)
