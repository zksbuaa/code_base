from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
from hnsw_fast import knn_search, load_index  # 假设 knn_search 和 load_index 已经在 hnsw_search.py 中实现
import json

# 创建 FastAPI 应用
app = FastAPI()

# 初始化 ：预加载 HNSW 索引
index_path_1 = "/mnt/public/code/dynamic_train/database/wangke/data/index_64_512_80M_03.bin"
index_path_2 = "/mnt/public/code/dynamic_train/database/wangke/data/index_64_512_80M_04.bin"

dim = 1024
hnsw1 = load_index(index_path_1, dim)
hnsw2 = load_index(index_path_2, dim)
print(f"hnsw1 type: {type(hnsw1)}, hnsw2 type: {type(hnsw2)}")

print("Indexes loaded.")

# 定义请求模型
class SearchRequest(BaseModel):
    query_indexes: list[int]  # 查询索引
    totalNum: int             # 数据库总数

@app.post("/knn_search/")
def search(request: SearchRequest):
    start_time = time.time()
    try:
        print(type(hnsw1))
        return_idx = knn_search(request.query_indexes, request.totalNum, hnsw1, hnsw2)
        return {"return_idx": return_idx, "execution_time": time.time() - start_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 运行应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)