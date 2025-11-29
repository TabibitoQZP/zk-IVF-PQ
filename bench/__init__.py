import numpy as np
from zk_IVF_PQ.zk_IVF_PQ import single_hash

MAX_VAL = 65535
MAX_DIS = 2**62 - 1


def cluster_gen(idx, n, M, K, rng=None):
    # 生成簇并计算簇哈希
    if rng is None:
        rng = np.random.default_rng()
    vpqs = rng.integers(0, K, size=(n, M), dtype=np.int64, endpoint=False)
    valid = rng.integers(0, 1, size=(n), dtype=np.int64, endpoint=True)
    items = rng.integers(0, MAX_VAL, size=(n), dtype=np.int64, endpoint=True)

    hash_list = []
    for i in range(n):
        left = np.array([idx, i, valid[i], items[i]], dtype=np.int64)
        hash_list.append(single_hash(np.concatenate([left, vpqs[i]])))

    while len(hash_list) > 1:
        hash_list = [
            single_hash([hash_list[2 * i], hash_list[2 * i + 1]])
            for i in range(len(hash_list) // 2)
        ]
    return vpqs, valid, items, hash_list[0]


def data_gen(D, n_list, M, K, d, n_probe, n):
    """
    统一的数据生成接口
    """
    rng = np.random.default_rng()

    query = rng.integers(0, MAX_VAL, size=(D,), dtype=np.int64, endpoint=True)
    ivf_center = rng.integers(
        0, MAX_VAL, size=(n_list, D), dtype=np.int64, endpoint=True
    )
    codebooks = rng.integers(0, MAX_VAL, size=(M, K, d), dtype=np.int64, endpoint=True)

    c = np.sum((ivf_center - query) ** 2, axis=1).astype(np.int64)
    order = np.argsort(c, kind="stable")
    order = order.astype(np.int64)
    cluster_idxes = order[:n_probe]

    vpqss = []
    valids = []
    itemss = []
    ivf_roots = []
    for i in range(n_list):
        vpqs, valid, items, curr_root = cluster_gen(i, n, M, K, rng)
        vpqss.append(vpqs)
        valids.append(valid)
        itemss.append(items)
        ivf_roots.append(curr_root)

    ivf_roots = np.array(ivf_roots, dtype=np.uint64)
    vpqss = np.stack([vpqss[i] for i in cluster_idxes], axis=0)
    valids = np.stack([valids[i] for i in cluster_idxes], axis=0)
    itemss = np.stack([itemss[i] for i in cluster_idxes], axis=0)

    return (
        query,
        ivf_center,
        cluster_idxes,
        vpqss,
        valids,
        itemss,
        codebooks,
        ivf_roots,
    )
