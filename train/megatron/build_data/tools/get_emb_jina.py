import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/mnt/public2/data/ssc/.hf_home"

from transformers import AutoTokenizer, AutoModel
from indexed_dataset import MMapIndexedDataset
import torch
import argparse
import math
import time
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_model_name", type=str, default="/mnt/public/model/huggingface/bge-m3")
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument("--dataset_path", type=str, default="/mnt/public/code/dynamic_train/dt-data/more_math_code_2_schema")
    parser.add_argument("--input_tokenizer_path", type=str, default="/mnt/public/code/data_mixture/models/Qwen2.5-0.5B")
    parser.add_argument("--save_emb_path", type=str, default="/mnt/public/code/dynamic_train/dt-data/emb-cluster/data")

    parser.add_argument("--total_docs", type=int, required=True)
    parser.add_argument("--total_nodes", type=int, required=True)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument("--total_gpus", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    return parser.parse_args()


def read_memmap_dataset(start_idx, end_idx, dataset, input_tokenizer) -> list[str]:
    tokens = dataset[start_idx:end_idx]
    texts = [input_tokenizer.decode(token) for token in tokens]
    return texts


if __name__ == "__main__":
    args = get_args()

    emb_model = AutoModel.from_pretrained(args.emb_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    input_dataset = MMapIndexedDataset(args.dataset_path, skip_warmup=True)
    input_tokenizer = AutoTokenizer.from_pretrained(args.input_tokenizer_path)

    docs_per_gpu = math.ceil(args.total_docs / args.total_gpus)
    gpu_rank = args.rank * args.gpus_per_node + args.gpu_id
    
    # 计算要处理的doc的start和end
    start_idx = gpu_rank * docs_per_gpu
    if start_idx >= args.total_docs:
        print(f"Rank {args.rank} GPU {args.gpu_id} has no data to process.")
        exit(0)
    end_idx = start_idx + docs_per_gpu
    if end_idx > args.total_docs:
        end_idx = args.total_docs
    
    print(f"Rank {args.rank} GPU {args.gpu_id} processing docs from {start_idx} to {end_idx}")

    # 保存到一个 numpy memmap格式的split中
    emb_cache_path = args.save_emb_path
    split_path = os.path.join(emb_cache_path, f"emb_split_{args.rank}_{args.gpu_id}.npy")
    if not os.path.exists(emb_cache_path):
        os.makedirs(emb_cache_path, exist_ok=True)
    if os.path.exists(split_path):
        emb_split = np.memmap(split_path, mode='r+', dtype=np.float32, shape=(end_idx - start_idx, args.emb_dim))
    else:
        emb_split = np.memmap(split_path, mode='w+', dtype=np.float32, shape=(end_idx - start_idx, args.emb_dim))
    
    # 分batch读取数据并转换为embedding
    for i in tqdm(range(start_idx, end_idx, args.batch_size), desc=f"Rank {args.rank:02} GPU {args.gpu_id:02} processing"):
        micro_start_idx = i
        micro_end_idx = min(i + args.batch_size, end_idx)
        revised_batch_size = micro_end_idx - micro_start_idx

        # 判断下，如果emb_split中已经有数据了，就不需要重新计算了，仅当emb_split中batch的最后一个embedding为0时才需要重新计算
        if np.all(emb_split[i - start_idx + revised_batch_size - 1] == 0):
            batch_texts = read_memmap_dataset(micro_start_idx, micro_end_idx, input_dataset, input_tokenizer)
            batch_embedding = emb_model.encode(batch_texts, truncate_dim=args.emb_dim, max_length=args.max_seq_len)
            emb_split[i - start_idx : i - start_idx + revised_batch_size] = batch_embedding
        emb_split.flush()

    emb_split.flush()
    print(f"Rank {args.rank} GPU {args.gpu_id} finished processing docs from {start_idx} to {end_idx} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
