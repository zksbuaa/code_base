import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

from transformers import AutoTokenizer

# 全局变量，供子进程使用（必须在顶层定义）
_tokenizer = None
_model_name = None


def init_worker(model_name):
    """初始化每个进程的 tokenizer"""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_tokens_per_example(example):
    """对单个样本统计 human / gpt 的 token 数"""
    global _tokenizer
    local_counts = {"human": [], "gpt": []}

    for message in example.get("conversations", []):
        role = message.get("from")
        content = message.get("value", "")
        if role in {"human", "gpt"}:
            tokens = _tokenizer.encode(content, add_special_tokens=False)
            local_counts[role].append(len(tokens))

    return local_counts


def load_conversations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parallel_tokenize(data, model_name, num_workers):
    token_counts = {"human": [], "gpt": []}

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(model_name,)) as pool:
        results = list(tqdm(pool.imap(count_tokens_per_example, data), total=len(data), desc="Tokenizing"))

    for result in results:
        for role in ["human", "gpt"]:
            token_counts[role].extend(result[role])

    return token_counts


def plot_distribution(token_counts, output_path=None):
    plt.figure(figsize=(10, 5))

    for role, counts in token_counts.items():
        print(f"{role} sum: {sum(counts)}, count: {len(counts)}, mean: {sum(counts)/len(counts) if counts else 0:.1f}, max: {max(counts) if counts else 0}")

    for role, counts in token_counts.items():
        if counts:
            plt.hist(counts, bins=50, alpha=0.6, label=f'{role} (mean={sum(counts)/len(counts):.1f})')

    plt.xlabel("Token count")
    plt.ylabel("Frequency")
    plt.title("Token Count Distribution per Role")
    plt.legend()
    plt.grid(True)
    

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Multiprocessing token count statistics for ShareGPT format.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the ShareGPT JSON file")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name for tokenizer")
    parser.add_argument("--output_plot", type=str, default=None, help="Optional path to save the token plot")
    parser.add_argument("--workers", type=int, default=8, help="Number of processes to use")

    args = parser.parse_args()

    print(f"Loading dataset from {args.json_path} ...")
    data = load_conversations(args.json_path)

    print(f"Starting tokenization with {args.workers} processes...")
    token_counts = parallel_tokenize(data, args.model_name, args.workers)

    print("Plotting results...")
    plot_distribution(token_counts, args.output_plot)


if __name__ == "__main__":
    # Required for Windows compatibility
    mp.set_start_method("spawn", force=True)
    main()


'''
python sharegpt_token_stats.py \
  --json_path my_data.json \
  --model_name Qwen/Qwen1.5-1.8B \
  --workers 16 \
  --output_plot token_distribution.png
'''