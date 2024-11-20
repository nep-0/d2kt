import numpy as np
import faiss

faiss_read_index = faiss.read_index('faiss_index.index')

# 假设索引是 1024 维的
query_vector = np.random.rand(1024).astype("float32")

# 查询前5个相似向量
try:
    distances, indices = faiss_read_index.search(np.array([query_vector]).astype('float32'), k=2)
    print(f"Indices of nearest neighbors: {indices}")
    print(f"Distances: {distances}")
except Exception as e:
    print(f"Query failed: {e}")
