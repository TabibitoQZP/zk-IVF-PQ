from vec_data_load.sift import SIFT
from bench.acc_bench import run_accuracy_bench

sift = SIFT("data/siftsmall/")
summary = run_accuracy_bench(
    sift.base_vecs,
    sift.query_vecs,
    top_k=10,
    name="siftsmall_default",
    n_list=64,
    M=8,
    K=256,
    n_probe=8,
    num_runs=5,
)
print(summary)
