import faiss, numpy as np

print("faiss module:", faiss.__file__)
print("has StandardGpuResources:", hasattr(faiss, "StandardGpuResources"))
try:
    print("num_gpus (faiss):", faiss.get_num_gpus())
except Exception as e:
    print("get_num_gpus() unavailable:", e)

# 做一次最小 GPU 操作自测（如果是 CPU 版会报错或退回 CPU）
D = 32
idx = faiss.IndexFlatL2(D)
x = np.random.randn(1000, D).astype("float32")
idx.add(x)
try:
    res = faiss.StandardGpuResources()  # 如果这里报 AttributeError，多半是 CPU 版
    idx_gpu = faiss.index_cpu_to_gpu(res, 0, idx)
    faiss_idx_kind = type(idx_gpu).__name__
    print("moved to GPU ok:", faiss_idx_kind)
except Exception as e:
    print("GPU move failed:", e)
