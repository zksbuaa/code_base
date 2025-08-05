import numpy as np
from indexed_dataset import MMapIndexedDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="/mnt/public/code/dynamic_train/dt-data/more_math_code_2_schema", help='dataset path（不带 .bin / .idx 后缀）')
    parser.add_argument('--sample-index', type=int, default=None, help='sample index to show')
    parser.add_argument('--tokenizer-path', type=str, default="/mnt/public/code/data_mixture/models/Qwen2.5-0.5B", help='tokenizer path')
    args = parser.parse_args()

    # 1. 指定预处理后数据集路径（不带 .bin / .idx 后缀）
    #    假设存在 "my_dataset.bin" 和 "my_dataset.idx"
    DATASET_PATH = args.dataset_path
    print(f"Dataset path: {DATASET_PATH}")

    # 2. 初始化 MMapIndexedDataset
    #    skip_warmup=True 可以减少加载时间（尤其数据集较大时）
    ds = MMapIndexedDataset(DATASET_PATH, skip_warmup=True)

    # 3. 获取数据集的样本数
    num_samples = len(ds)
    print(f"Number of samples: {num_samples}")

    # 4. 快速统计所有样本的 Token 总数
    #    ds.sizes 记录了每条样本（即一段token序列）的长度
    #    直接求和即可得到总 token 数
    total_tokens = ds.sizes.sum()
    print(f"Total token count: {total_tokens}")

    # 5. 演示：读取某个特定下标（index）的样本 Token ID
    sample_index = args.sample_index
    if sample_index != None and sample_index < num_samples:
        token_ids = ds[sample_index]
        print(f"Token IDs of sample {sample_index}: {token_ids}")
        print(f"Length of sample {sample_index}: {len(token_ids)} tokens")

    # 6. 演示：使用 Hugging Face Transformers 加载预训练模型和 Tokenizer
    if sample_index != None and sample_index < num_samples:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        seq = tokenizer.decode(token_ids)
        print(f"Decoded sequence: {seq}")


if __name__ == "__main__":
    main()
