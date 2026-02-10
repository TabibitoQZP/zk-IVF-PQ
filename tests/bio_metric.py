from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bench.bio_metric import (
    cache_bupt_cbface_topk,
    hit_curve_from_cache,
    recall_curve_from_cache,
)


TOPK_CACHE_DIR = Path("data") / "bio_metric" / "topk_cache"


def _load_topk_cache_meta(path: str | Path) -> Dict[str, object]:
    p = Path(path)
    with np.load(str(p), allow_pickle=False) as data:
        return json.loads(data["meta_json"].item())


def _ci95_mean(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0)
    if arr.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    std = arr.std(axis=0, ddof=1)
    ci = 1.96 * std / math.sqrt(float(arr.shape[0]))
    return mean, ci


def ensure_bupt_cbface_topk_caches(
    dataset_dir: str | Path,
    *,
    num_runs: int,
    top_k: int = 100,
    name: str | None = None,
    num_queries: int = 1024,
    ground_truth_k: int = 11,
    sample_seed: int = 0,
    db_name: str = "arcface_embeddings.duckdb",
    n_list: int = 1024,
    M: int = 8,
    K: int = 256,
    n_probe: int = 8,
    scale_n: int = 65536,
    cluster_bound: int | None = 1024,
    cache_dir: str | Path = TOPK_CACHE_DIR,
) -> List[Path]:
    """
    对同一配置确保有 `num_runs` 个 topk-cache `.npz`：
      - 如果已有数量足够：直接返回已有的 cache 路径（不重新训练/检索）
      - 如果不够：自动补齐到 num_runs

    这里“同一配置”默认固定 `sample_seed`，不同 run 用不同 `train_seed` 区分。
    """
    if num_runs <= 0:
        raise ValueError("num_runs must be > 0")

    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"{dataset_dir.name}_seed{sample_seed}"

    expected: Dict[str, object] = {
        "name": str(name),
        "sample_seed": int(sample_seed),
        "ground_truth_k": int(ground_truth_k),
        "db_name": str(db_name),
        "n_list": int(n_list),
        "M": int(M),
        "K": int(K),
        "n_probe": int(n_probe),
        "top_k": int(top_k),
        "scale_n": int(scale_n),
        "cluster_bound": None if cluster_bound is None else int(cluster_bound),
    }

    existing: List[Tuple[int, Path]] = []
    for p in sorted(cache_dir.glob("topk_*.npz")):
        if f"_Q{num_queries}_" not in p.name:
            continue
        try:
            meta = _load_topk_cache_meta(p)
        except Exception:
            continue
        if any(meta.get(k) != v for k, v in expected.items()):
            continue
        train_seed = int(meta.get("train_seed", -1))
        existing.append((train_seed, p))

    existing.sort(key=lambda x: x[0])
    if len(existing) >= num_runs:
        return [p for _, p in existing[:num_runs]]

    have_train_seeds = {s for s, _ in existing}
    out = [p for _, p in existing]

    train_seed = 0
    while len(out) < num_runs:
        while train_seed in have_train_seeds:
            train_seed += 1

        p = cache_bupt_cbface_topk(
            dataset_dir,
            top_k=top_k,
            name=name,
            num_queries=num_queries,
            ground_truth_k=ground_truth_k,
            sample_seed=sample_seed,
            train_seed=train_seed,
            db_name=db_name,
            n_list=n_list,
            M=M,
            K=K,
            n_probe=n_probe,
            scale_n=scale_n,
            cluster_bound=cluster_bound,
            cache_path=None,
            overwrite=False,
        )

        meta = _load_topk_cache_meta(p)
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Cache config mismatch: {p}")
        if int(meta.get("train_seed", -1)) != int(train_seed):
            raise ValueError(f"Cache train_seed mismatch: {p}")
        if f"_Q{num_queries}_" not in Path(p).name:
            raise ValueError(f"Cache num_queries mismatch: {p}")

        out.append(Path(p))
        have_train_seeds.add(train_seed)
        train_seed += 1

    return out


def ci95_from_topk_caches(paths: List[str | Path]) -> Dict[str, np.ndarray]:
    """
    从多个 `.npz` cache 直接计算 mean ± 95%CI（跨 run）。

    返回：
      - standard_mean_hit_curve / standard_ci95_hit_curve
      - standard_mean_recall_curve / standard_ci95_recall_curve
      - zk_mean_hit_curve / zk_ci95_hit_curve
      - zk_mean_recall_curve / zk_ci95_recall_curve
    """
    if not paths:
        raise ValueError("paths must be non-empty")

    std_hit: List[np.ndarray] = []
    zk_hit: List[np.ndarray] = []
    std_recall: List[np.ndarray] = []
    zk_recall: List[np.ndarray] = []

    for p in paths:
        hit_curves = hit_curve_from_cache(p)
        recall_curves = recall_curve_from_cache(p)
        std_hit.append(np.asarray(hit_curves["standard_hit_curve"], dtype=np.float64))
        zk_hit.append(np.asarray(hit_curves["zk_hit_curve"], dtype=np.float64))
        std_recall.append(
            np.asarray(recall_curves["standard_recall_curve"], dtype=np.float64)
        )
        zk_recall.append(np.asarray(recall_curves["zk_recall_curve"], dtype=np.float64))

    std_hit_arr = np.stack(std_hit, axis=0)
    zk_hit_arr = np.stack(zk_hit, axis=0)
    std_recall_arr = np.stack(std_recall, axis=0)
    zk_recall_arr = np.stack(zk_recall, axis=0)

    std_hit_mean, std_hit_ci = _ci95_mean(std_hit_arr)
    zk_hit_mean, zk_hit_ci = _ci95_mean(zk_hit_arr)
    std_recall_mean, std_recall_ci = _ci95_mean(std_recall_arr)
    zk_recall_mean, zk_recall_ci = _ci95_mean(zk_recall_arr)

    return {
        "standard_mean_hit_curve": std_hit_mean,
        "standard_ci95_hit_curve": std_hit_ci,
        "standard_mean_recall_curve": std_recall_mean,
        "standard_ci95_recall_curve": std_recall_ci,
        "zk_mean_hit_curve": zk_hit_mean,
        "zk_ci95_hit_curve": zk_hit_ci,
        "zk_mean_recall_curve": zk_recall_mean,
        "zk_ci95_recall_curve": zk_recall_ci,
    }


if __name__ == "__main__":
    avgval = 50
    caches = ensure_bupt_cbface_topk_caches(
        f"data/BUPT-CBFace-{avgval}",
        ground_truth_k=avgval - 1,
        top_k=100,
        sample_seed=0,
        cluster_bound=1024,
        num_runs=5,
    )
    summary = ci95_from_topk_caches(caches)
    for k in (1, 10, 100):
        i = k - 1
        print(
            f"k={k:3d} "
            f"std_hit={summary['standard_mean_hit_curve'][i]:.6f}±{summary['standard_ci95_hit_curve'][i]:.6f} "
            f"std_recall={summary['standard_mean_recall_curve'][i]:.6f}±{summary['standard_ci95_recall_curve'][i]:.6f} "
            f"zk_hit={summary['zk_mean_hit_curve'][i]:.6f}±{summary['zk_ci95_hit_curve'][i]:.6f} "
            f"zk_recall={summary['zk_mean_recall_curve'][i]:.6f}±{summary['zk_ci95_recall_curve'][i]:.6f}"
        )
    print()

    avgval = 12
    caches = ensure_bupt_cbface_topk_caches(
        f"data/BUPT-CBFace-{avgval}",
        ground_truth_k=avgval - 1,
        top_k=100,
        sample_seed=0,
        cluster_bound=1024,
        num_runs=5,
    )
    summary = ci95_from_topk_caches(caches)
    for k in (1, 10, 100):
        i = k - 1
        print(
            f"k={k:3d} "
            f"std_hit={summary['standard_mean_hit_curve'][i]:.6f}±{summary['standard_ci95_hit_curve'][i]:.6f} "
            f"std_recall={summary['standard_mean_recall_curve'][i]:.6f}±{summary['standard_ci95_recall_curve'][i]:.6f} "
            f"zk_hit={summary['zk_mean_hit_curve'][i]:.6f}±{summary['zk_ci95_hit_curve'][i]:.6f} "
            f"zk_recall={summary['zk_mean_recall_curve'][i]:.6f}±{summary['zk_ci95_recall_curve'][i]:.6f}"
        )
    print()

    avgval = 50
    caches = ensure_bupt_cbface_topk_caches(
        f"data/BUPT-CBFace-{avgval}",
        ground_truth_k=avgval - 1,
        top_k=100,
        sample_seed=0,
        n_list=8192,
        n_probe=64,
        cluster_bound=128,
        num_runs=5,
    )
    summary = ci95_from_topk_caches(caches)
    for k in (1, 10, 100):
        i = k - 1
        print(
            f"k={k:3d} "
            f"std_hit={summary['standard_mean_hit_curve'][i]:.6f}±{summary['standard_ci95_hit_curve'][i]:.6f} "
            f"std_recall={summary['standard_mean_recall_curve'][i]:.6f}±{summary['standard_ci95_recall_curve'][i]:.6f} "
            f"zk_hit={summary['zk_mean_hit_curve'][i]:.6f}±{summary['zk_ci95_hit_curve'][i]:.6f} "
            f"zk_recall={summary['zk_mean_recall_curve'][i]:.6f}±{summary['zk_ci95_recall_curve'][i]:.6f}"
        )
    print()

    avgval = 12
    caches = ensure_bupt_cbface_topk_caches(
        f"data/BUPT-CBFace-{avgval}",
        ground_truth_k=avgval - 1,
        top_k=100,
        sample_seed=0,
        n_list=8192,
        n_probe=64,
        cluster_bound=128,
        num_runs=5,
    )
    summary = ci95_from_topk_caches(caches)
    for k in (1, 10, 100):
        i = k - 1
        print(
            f"k={k:3d} "
            f"std_hit={summary['standard_mean_hit_curve'][i]:.6f}±{summary['standard_ci95_hit_curve'][i]:.6f} "
            f"std_recall={summary['standard_mean_recall_curve'][i]:.6f}±{summary['standard_ci95_recall_curve'][i]:.6f} "
            f"zk_hit={summary['zk_mean_hit_curve'][i]:.6f}±{summary['zk_ci95_hit_curve'][i]:.6f} "
            f"zk_recall={summary['zk_mean_recall_curve'][i]:.6f}±{summary['zk_ci95_recall_curve'][i]:.6f}"
        )
    print()

    avgval = 50
    caches = ensure_bupt_cbface_topk_caches(
        f"data/BUPT-CBFace-{avgval}",
        ground_truth_k=avgval - 1,
        top_k=100,
        sample_seed=0,
        n_list=512,
        n_probe=4,
        cluster_bound=2048,
        num_runs=5,
    )
    summary = ci95_from_topk_caches(caches)
    for k in (1, 10, 100):
        i = k - 1
        print(
            f"k={k:3d} "
            f"std_hit={summary['standard_mean_hit_curve'][i]:.6f}±{summary['standard_ci95_hit_curve'][i]:.6f} "
            f"std_recall={summary['standard_mean_recall_curve'][i]:.6f}±{summary['standard_ci95_recall_curve'][i]:.6f} "
            f"zk_hit={summary['zk_mean_hit_curve'][i]:.6f}±{summary['zk_ci95_hit_curve'][i]:.6f} "
            f"zk_recall={summary['zk_mean_recall_curve'][i]:.6f}±{summary['zk_ci95_recall_curve'][i]:.6f}"
        )
    print()

    avgval = 12
    caches = ensure_bupt_cbface_topk_caches(
        f"data/BUPT-CBFace-{avgval}",
        ground_truth_k=avgval - 1,
        top_k=100,
        sample_seed=0,
        n_list=512,
        n_probe=4,
        cluster_bound=2048,
        num_runs=5,
    )
    summary = ci95_from_topk_caches(caches)
    for k in (1, 10, 100):
        i = k - 1
        print(
            f"k={k:3d} "
            f"std_hit={summary['standard_mean_hit_curve'][i]:.6f}±{summary['standard_ci95_hit_curve'][i]:.6f} "
            f"std_recall={summary['standard_mean_recall_curve'][i]:.6f}±{summary['standard_ci95_recall_curve'][i]:.6f} "
            f"zk_hit={summary['zk_mean_hit_curve'][i]:.6f}±{summary['zk_ci95_hit_curve'][i]:.6f} "
            f"zk_recall={summary['zk_mean_recall_curve'][i]:.6f}±{summary['zk_ci95_recall_curve'][i]:.6f}"
        )
    print()
