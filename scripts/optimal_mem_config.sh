# 这个乱写的
python -m bench.optimal_mem_config \
  --N 65536 \
  --D 128 \
  --n-list 256 \
  --n-probe 16 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 这个是针对8192固定selection的配置
python -m bench.optimal_mem_config \
  --N 65536 \
  --D 256 \
  --n-list 512 \
  --n-probe 64 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 这个是针对SIFT的配置, 选了2048个...
python -m bench.optimal_mem_config \
  --N 2097152 \
  --D 128 \
  --n-list 8192 \
  --n-probe 8 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 这个是针对SIFT的配置, 选了16384个...
python -m bench.optimal_mem_config \
  --N 2097152 \
  --D 128 \
  --n-list 8192 \
  --n-probe 64 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 这个是针对SIFT的配置, 选了16384个, 但粗分簇
python -m bench.optimal_mem_config \
  --N 2097152 \
  --D 128 \
  --n-list 1024 \
  --n-probe 8 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 针对GIST的配置, 选了2048个...
python -m bench.optimal_mem_config \
  --N 2097152 \
  --D 960 \
  --n-list 8192 \
  --n-probe 8 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 针对GIST的配置, 选了16384个...
python -m bench.optimal_mem_config \
  --N 2097152 \
  --D 960 \
  --n-list 8192 \
  --n-probe 64 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5

# 这个是针对GIST的配置, 选了16384个, 但粗分簇
python -m bench.optimal_mem_config \
  --N 2097152 \
  --D 960 \
  --n-list 1024 \
  --n-probe 8 \
  --mem-bits 64 \
  --c 10 \
  --num-runs 5