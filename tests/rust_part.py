import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import single_hash

if __name__ == "__main__":
    hash_val = single_hash(np.array([22, 1], dtype=np.int32))
    print(hash_val)
