import time
from vec_data_load.sift import SIFT
from bench.acc_bench import run_accuracy_bench

stime = time.time()
sift = SIFT("data/sift/")
print(sift.base_vecs.shape)
print(sift.query_vecs.shape)
print(sift.gt_vecs.shape)
summary = run_accuracy_bench(
    sift.base_vecs,
    sift.query_vecs,
    sift.gt_vecs,
    top_k=100,
    name="sift_1m-withgt",
    n_list=1024,
    M=8,
    K=256,
    n_probe=8,
    num_runs=1,
    cluster_bound=2048,
)
print(summary)
print((time.time() - stime) / 3600)
