import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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


def get_embedding(text: list[str], tokenizer, model, max_seq_len) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.cpu().numpy().astype(np.float32)


def read_memmap_dataset(start_idx, end_idx, dataset, input_tokenizer) -> list[str]:
    tokens = dataset[start_idx:end_idx]
    texts = [input_tokenizer.decode(token) for token in tokens]
    return texts

if __name__ == "__main__":
    args = get_args()
    emb_tokenizer = AutoTokenizer.from_pretrained(args.emb_model_name, use_fast=True)
    emb_model = AutoModel.from_pretrained(args.emb_model_name).to("cuda").eval()

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
    log_interval = 0
    for i in tqdm(range(start_idx, end_idx, args.batch_size), disable=(args.gpu_id != 0)):
        micro_start_idx = i
        micro_end_idx = min(i + args.batch_size, end_idx)
        revised_batch_size = micro_end_idx - micro_start_idx

        if np.all(emb_split[i - start_idx + revised_batch_size - 1] == 0):
            batch_texts = read_memmap_dataset(micro_start_idx, micro_end_idx, input_dataset, input_tokenizer)
            batch_embedding = get_embedding(batch_texts, emb_tokenizer, emb_model, args.max_seq_len)
            emb_split[i - start_idx : i - start_idx + revised_batch_size] = batch_embedding
        
        if log_interval % 10 == 0:
            emb_split.flush()
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Rank {args.rank} GPU {args.gpu_id}: {micro_end_idx} / {end_idx} docs ({(micro_end_idx-start_idx)/(end_idx-start_idx):.2%})", flush=True)
        log_interval += 1

    emb_split.flush()
    print(f"Rank {args.rank} GPU {args.gpu_id} finished processing docs from {start_idx} to {end_idx} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
