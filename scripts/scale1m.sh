# python bench_free_bench/brute_force.py --N 1048576 --D 128
# python bench_free_bench/pq_flat_verify.py --N 1048576 --D 128 --M 8 --K 256
python bench_free_bench/ivf_flat.py --N 1048576 --d 128 --n_list 1024 --n_probe 16
python bench_free_bench/ivf_pq_verify.py --N 1048576 --D 128 --n_list 1024 --n_probe 16 --M 8 --K 256
