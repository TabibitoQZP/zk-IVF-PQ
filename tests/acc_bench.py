from vec_data_load.sift import SIFT
from bench.acc_bench import run_accuracy_bench

sift = SIFT("data/gist/")
print(sift.base_vecs.shape)
summary = run_accuracy_bench(
    sift.base_vecs,
    sift.query_vecs,
    top_k=10,
    name="gist_1m",
    n_list=1024,
    M=30,
    K=256,
    n_probe=8,
    num_runs=5,
    cluster_bound=2048,
)
print(summary)
