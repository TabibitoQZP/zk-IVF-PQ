# maturin develop --release

# # 乱写的, 无意味
# python -m bench.optimal_config \
#   --N 65536 \
#   --D 128 \
#   --M 8 \
#   --K 16 \
#   --selected_count 8192 \
#   --c 13 \
#   --num-runs 5
#
# # 64bit的配置
# python -m bench.optimal_config \
#   --N 65536 \
#   --D 256 \
#   --M 8 \
#   --K 256 \
#   --selected_count 8192 \
#   --c 8 \
#   --num-runs 5
#
# # 乱写的
# python -m bench.optimal_config \
#   --N 1048576 \
#   --D 128 \
#   --M 8 \
#   --K 16 \
#   --selected_count 8192 \
#   --c 10 \
#   --num-runs 5
#
# # 乱写的
# python -m bench.optimal_config \
#   --N 1048576 \
#   --D 1024 \
#   --M 32 \
#   --K 256 \
#   --selected_count 8192 \
#   --c 6 \
#   --num-runs 5
#
# # SIFT在16384选取下的配置
# python -m bench.optimal_config \
#   --N 2097152 \
#   --D 128 \
#   --M 8 \
#   --K 256 \
#   --selected_count 16384 \
#   --c 7 \
#   --num-runs 5
#
python -m bench.optimal_config \
  --N 2097152 \
  --D 128 \
  --M 8 \
  --K 256 \
  --selected_count 16384 \
  --c 9 \
  --num-runs 5

# GIST在16384选取下的配置
python -m bench.optimal_config \
  --N 2097152 \
  --D 960 \
  --M 8 \
  --K 256 \
  --selected_count 16384 \
  --c 7 \
  --num-runs 5
