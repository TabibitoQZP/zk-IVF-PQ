use crate::hash_gadgets::hash_u64;
use crate::utils::metrics::measure_memory_usage;
use std::time::Instant;

pub fn log2(mut n: usize) -> usize {
    let mut result: usize = 0;
    while n > 1 {
        n /= 2;
        result += 1;
    }
    result
}

pub fn merkle_run(n_list: usize, n: usize, M: usize, loop_time: usize) -> (f64, u64) {
    let (duration, memory_peak) = measure_memory_usage(|| {
        let log_n_list = log2(n_list);
        let log_n = log2(n);
        let mut ivf_tree: Vec<u64> = vec![0; log_n_list + 1];
        let mut cluster_tree: Vec<u64> = vec![0; log_n + 1];
        let mut random_val: u64 = 0;

        let now = Instant::now();
        for _ in 0..loop_time {
            let leaf_hash = hash_u64(vec![random_val; M + 4]);
            cluster_tree[0] = leaf_hash;
            for i in 0..log_n {
                cluster_tree[i + 1] = hash_u64(vec![cluster_tree[i], cluster_tree[i + 1]]);
            }
            ivf_tree[0] = cluster_tree[log_n];
            for i in 0..log_n_list {
                ivf_tree[i + 1] = hash_u64(vec![ivf_tree[i], ivf_tree[i + 1]]);
            }
            random_val = ivf_tree[log_n_list];
        }
        let duration = now.elapsed().as_secs_f64();
        duration / (loop_time as f64)
    });

    (duration, memory_peak)
}
