import os
import numpy as np
import argparse
from tqdm import tqdm
from itertools import product
from pprint import pprint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=1024, help="Dimension of embeddings")
    parser.add_argument("--save_emb_path", type=str, default="/mnt/public/code/dynamic_train/dt-data/emb-cluster/data/.emb_splits", help="Path to the directory containing emb_splits")
    parser.add_argument("--output_file", type=str, default="/mnt/public/code/dynamic_train/dt-data/emb-cluster/data/merged_embs.npy", help="Path to the final merged .npy file")
    parser.add_argument("--total_docs", type=int, default=163615821, help="Total number of documents")
    parser.add_argument("--total_nodes", type=int, default=15, help="Total number of nodes")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="Number of GPUs per node")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    split_files = [
        os.path.join(args.save_emb_path, f"emb_split_{node_id}_{gpu_id}.npy")
        for node_id, gpu_id
        in product(range(args.total_nodes), range(args.gpus_per_node))
    ]
    
    for split_file in split_files:
        assert os.path.exists(split_file), f"Split file {split_file} does not exist"
    
    print(f"Found {len(split_files)} split files:")
    pprint(split_files)
    
    # 创建最终的 memmap 文件
    merged_emb = np.memmap(args.output_file, mode='w+', dtype=np.float32, shape=(args.total_docs, args.emb_dim))
    
    current_index = 0
    for split_file in tqdm(split_files, desc="Merging splits"):
        emb_data = np.memmap(split_file, mode='r', dtype=np.float32).reshape(-1, args.emb_dim)
        num_embeddings = emb_data.shape[0]
        print(f"Adding {num_embeddings} embeddings from {split_file} to index {current_index}")
        
        merged_emb[current_index:current_index + num_embeddings] = emb_data
        current_index += num_embeddings
    
    merged_emb.flush()
    print(f"Merging completed. Final file saved at {args.output_file}")
