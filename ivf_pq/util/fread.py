import numpy as np


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.dtype("<i4"))
    if a.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    d = int(a[0])
    if d <= 0:
        raise ValueError(f"Bad dimension {d} in {path}")
    if a.size % (d + 1) != 0:
        raise ValueError("File size is not a multiple of (d+1) int32 words")

    a = a.reshape(-1, d + 1)
    if not np.all(a[:, 0] == d):
        raise ValueError("Inconsistent per-vector dimension headers")

    vecs = a[:, 1:].view(np.dtype("<f4")).copy()
    return vecs


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.dtype("<i4"))
    if a.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    d = int(a[0])
    if a.size % (d + 1) != 0:
        raise ValueError("Bad ivecs file length")
    a = a.reshape(-1, d + 1)
    if not np.all(a[:, 0] == d):
        raise ValueError("Inconsistent per-vector dimension headers")
    return a[:, 1:].copy()
