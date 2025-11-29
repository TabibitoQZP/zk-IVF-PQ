maturin develop --release
python -m bench.set_based --N 1048576 --D 128 --M 8 --K 256 --n_list 1024 --n_probe 8
