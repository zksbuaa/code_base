import os, sys, time
import platform
import resource
import numpy as np
import hnswlib

np.set_printoptions(threshold = 1024)

def memory():
    # 获取当前进程的最大驻留集大小（以字节为单位）
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # 注意：在Linux上返回的是KB，在macOS上是字节
    if os.name == 'posix' and platform.system() == 'Linux':
        print(f"Max RSS: {max_rss / (1024 * 1024):.3f} GB")
    else:
        print(f"Max RSS: {max_rss / (1024 * 1024 * 1024):.3f} GB")


def mainT():
    total_num = 163615821
    dim = 1024
    query_num = 10000
    insert_num = 80000000
    batch_size = 3000000
    get_data_num = insert_num + query_num

    index_path = "/mnt/public/code/dynamic_train/database/wangke/data/index_64_512_80M_03.bin"
    data_path = "/mnt/public/code/dynamic_train/dt-data/merged_embs.npy"
    data = np.memmap(data_path, dtype='float32', mode='r', shape=(get_data_num, dim))  
    shape = data.shape
    print(shape[0])
    print(shape[1])

    func_start_time = time.time()
    hnsw = hnswlib.Index(space='l2', dim=dim)
    hnsw.init_index(max_elements=get_data_num, ef_construction=512, M=64)
    hnsw.set_ef(512)
    hnsw.set_num_threads(90)

    data_insert = data[0: insert_num]
    ids = np.arange(len(data_insert))

    print("data_insert size=%s" % len(data_insert))
    print(f"初始索引 time :{time.time()-func_start_time} s")

    func_start_time = time.time()
    # 共有多少批次（最后一个批次可能不足batch_size）
    num_batches = (insert_num + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, insert_num)
        print(f"往索引写入数据 i={i},  start_index={start_index},  end_index={end_index}")
        hnsw.add_items(data_insert[start_index:end_index], ids[start_index:end_index])        
        print(f"batch写入时间 i, time : {i}, {time.time()-func_start_time} s")

    # print(f"往索引写入数据")
    print(f"往索引写入数据总时间 time :{time.time()-func_start_time} s")
    memory()
    
    data_query = data[get_data_num - 1 - query_num: get_data_num - 1]
    print("data_query size=%s" % len(data_query))

    func_start_time = time.time()
    hnsw.set_ef(512)
    labels, distances = hnsw.knn_query(data_query, k = 500)
    print(f"查询时间 time :{time.time()-func_start_time} s")
    print("\n")

    func_start_time = time.time()
    hnsw.save_index(index_path)
    print(f"保存索引 time :{time.time()-func_start_time} s")
    print("Saving end\n")
    del hnsw


if __name__ == "__main__":
    mainT()
