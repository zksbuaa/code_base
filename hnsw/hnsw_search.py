import os
import numpy as np
import hnswlib
from tqdm import tqdm
import time

def load_index(index_path, dim, ef_search=None):
    """加载 HNSW 索引，并设置 ef_construction 和 ef_search 参数"""
    start_time = time.time()
    print(f"Loading index from {index_path}, start time: {start_time}")  
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件 {index_path} 不存在！")
    hnsw = hnswlib.Index(space='l2', dim=dim)
    hnsw.load_index(index_path)
    print(f"Load index done {index_path}, cost time:", time.time() - start_time)
    return hnsw

def duplicate_removal(ids1, ids2, totalNum, window_size, top_k):
    """HNSW 查询去重
    该函数用于在 HNSW（Hierarchical Navigable Small World）检索过程中去除重复的查询结果。

    参数:
    ids1: List[List[int]] - 第一组检索结果，每个子列表代表一次查询的返回 ID 列表。
    ids2: List[List[int]] - 第二组检索结果，与 ids1 结构相同，可能来自不同的索引或方法。
    window_size: int - 每次处理的窗口大小，即每次加入去重集合的 ID 范围步长。
    top_k: int - 目标去重后的总 ID 数量，按查询的数量进行放大，即 `top_k * len(ids1)`。

    返回:
    Tuple[Set[int], int] - 返回去重后的 ID 集合，以及最终处理的窗口范围。
    """
    all_retrieved_ids = set()  # 存储去重后的 ID
    current_window = 0  # 当前处理的窗口范围
    while len(all_retrieved_ids) < totalNum and current_window < top_k:
        current_window += window_size
        print(f"current_window range {current_window - window_size} to {current_window}")
        # 遍历所有查询的检索结果，按窗口步长加入去重集合
        for i in range(len(ids1)):
            all_retrieved_ids.update(ids1[i][current_window - window_size: current_window])
            all_retrieved_ids.update(ids2[i][current_window - window_size: current_window])
        print("current len(all_retrieved_ids):", len(all_retrieved_ids))
    return all_retrieved_ids, current_window

def evaluate_hnsw_recall(query_vectors, hnsw1, hnsw2, totalNum, window_size, top_k=1000):
    """计算 HNSW 索引合并后的 Recall（批量查询版本）
    
    该函数用于评估两个 HNSW 索引合并后的查询召回率。它先分别查询两个 HNSW 索引，
    然后对查询结果进行去重，最后计算去重后的查询结果占所有查询返回结果的比例。

    参数:
    query_vectors: np.ndarray - 查询向量数组，每一行代表一个查询向量。
    hnsw1: hnswlib.Index - 第一个 HNSW 索引对象。
    hnsw2: hnswlib.Index - 第二个 HNSW 索引对象。
    window_size: int - 处理窗口大小，决定每轮去重时使用的 ID 数量。
    top_k: int - 每个查询返回的近邻数量，默认 1000。

    返回:
    Tuple[float, Set[int]] - 计算得到的去重率（duplicate_rate），以及去重后的 ID 集合。
    """
    evaluate_start_time = time.time()
    print("Running batch HNSW search...")
    # 进行批量查询，返回的是查询结果的 ID 和对应的距离
    # 参数：
    # - query_vectors: 需要查询的向量集合，形状为 (query_num, d)，其中 query_num 是查询数量，d 是向量维度
    # - k: 需要返回的最近邻数量，即 top_k
    # 返回：
    # - ids1: 形状为 (query_num, top_k) 的list，每一行存储了该查询向量在 hnsw1 索引中找到的 top_k 个最近邻 ID
    # - distances1: 形状为 (query_num, top_k) 的list，每一行存储了对应最近邻 ID 的距离（通常是 L2 欧几里得距离）
    ids1, distances1 = hnsw1.knn_query(query_vectors, k=top_k)
    ids2, distances2 = hnsw2.knn_query(query_vectors, k=top_k)
    print("Finish HNSW search, start to calculate recall...")
    # 对查询结果进行去重
    duplicate_removal_ids, current_window = duplicate_removal(ids1, ids2, totalNum, window_size, top_k)
    # 计算去重率：
    # duplicate_removal_ids 是最终去重后的 ID 数量，
    # 而 (current_window * 2 * len(query_vectors)) 是两个索引返回的总 ID 数量
    duplicate_rate = len(duplicate_removal_ids) / (current_window * 2 * len(query_vectors))
    print(f"evaluate_hnsw_recall finished, evaluate cost time: {time.time() - evaluate_start_time:.2f}s")
    return duplicate_rate, duplicate_removal_ids

if __name__ == "__main__":
    start_time = time.time()  # 记录程序开始时间
    # 设定查询相关参数
    query_num = 2000  # 需要查询的向量个数
    read_num = 163615821  # 数据集中存储的总向量数量
    # 计算需要检索的 top_k 值
    # totalNum 表示最终需要获取的向量数量，设定平均重复度 β = 0.08002
    totalNum = 10_000_000  
    # 计算 top_k，考虑平均重复度 β 和额外冗余因子 0.05
    top_k = int(read_num / query_num * (1 + 0.08002 + 0.05))  
    d = 1024  # 向量维度
    window_size = 100  # 处理窗口大小，每轮去重时使用的 ID 数量
    # 载入数据文件路径
    data_path = "/mnt/public/code/dynamic_train/dt-data/merged_embs.npy"
    # 载入查询索引的文件路径
    query_idx_path = "/mnt/public/code/dynamic_train/database/mayang/query/query_idx_2000.txt"
    # 使用 memmap 直接映射数据，避免一次性加载到内存
    data = np.memmap(data_path, dtype='float32', mode='r', shape=(read_num, d))  
    print("数据集形状:", data.shape)
    # 读取查询向量的索引，并从数据集中提取查询向量
    query_indexes = np.loadtxt(query_idx_path, dtype=int)
    query_vectors = data[query_indexes]
    print("查询向量 shape:", query_vectors.shape)
    # HNSW 索引文件路径
    index_path_1 = "/mnt/public/code/dynamic_train/database/wangke/data/index_64_512_80M_03.bin"
    index_path_2 = "/mnt/public/code/dynamic_train/database/wangke/data/index_64_512_80M_04.bin"
    # 获取查询向量的维度（应与索引的维度匹配）
    dim = query_vectors.shape[1]
    # 加载 HNSW 索引
    hnsw1 = load_index(index_path_1, dim)
    hnsw2 = load_index(index_path_2, dim)
    print("HNSW 索引加载完成，耗时:", time.time() - start_time)
    # 设置 HNSW 的搜索参数 ef_search，影响搜索时的回溯范围
    ef_search = 5000
    hnsw1.set_ef(ef_search)
    hnsw2.set_ef(ef_search)
    # 计算 HNSW 查询的召回率，并去重 .return_idx 为去重后的 ID 集合
    duplicate_rate, return_idx = evaluate_hnsw_recall(query_vectors, hnsw1, hnsw2, totalNum, window_size, top_k)
    # 输出去重后的 ID 数量和重复率
    print(f"总去重 ID 数量: {len(return_idx)} ,重复率: {1 - duplicate_rate}")
    # 计算总运行时间
    print("总耗时:", time.time() - start_time)

