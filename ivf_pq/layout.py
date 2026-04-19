from __future__ import annotations

import numpy as np


def normalize_layout(layout: str | None) -> str | None:
    if layout is None:
        return None

    normalized = str(layout).strip().lower()
    if normalized in ("", "none"):
        return None
    if normalized == "mod8":
        return normalized

    raise ValueError(f"Unsupported layout: {layout!r}")


def build_modulo_permutation(dim: int, stride: int = 8) -> np.ndarray:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if dim % stride != 0:
        raise ValueError(
            f"dim={dim} must be divisible by stride={stride} for modulo reorder"
        )

    return np.concatenate(
        [np.arange(offset, dim, stride, dtype=np.int64) for offset in range(stride)],
        axis=0,
    )


def layout_permutation(dim: int, layout: str | None) -> np.ndarray | None:
    normalized = normalize_layout(layout)
    if normalized is None:
        return None
    if normalized == "mod8":
        return build_modulo_permutation(dim, stride=8)
    raise ValueError(f"Unsupported layout after normalization: {normalized!r}")


def apply_layout(arr: np.ndarray, layout: str | None) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 0:
        raise ValueError("apply_layout expects an array with at least one dimension")

    perm = layout_permutation(int(data.shape[-1]), layout)
    if perm is None:
        return np.ascontiguousarray(data)
    return np.ascontiguousarray(data[..., perm])


def layout_suffix(layout: str | None) -> str:
    normalized = normalize_layout(layout)
    if normalized is None:
        return ""
    return f"_layout{normalized}"
