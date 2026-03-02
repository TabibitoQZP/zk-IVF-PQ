import argparse
import time
from pathlib import Path

from vec_data_load.sift import SIFT
from bench.acc_bench import run_accuracy_bench
from ivf_pq import MAX_SCALE


def main(
    data_name: str,
    n_list: int,
    n_probe: int,
    cluster_bound: int | None,
    *,
    data_root: str = "data",
    M: int = 8,
    K: int = 256,
    top_k: int = 100,
    num_runs: int = 1,
    scale_n: int | None = None,
    force_recompute: bool = False,
    name: str | None = None,
):
    stime = time.time()
    data_dir = Path(data_root) / data_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {data_dir}")

    sift = SIFT(str(data_dir))
    print(data_name)
    print(sift.base_vecs.shape)
    print(sift.query_vecs.shape)
    print(sift.gt_vecs.shape)

    if name is None:
        name = f"NEW_{data_name}"

    if scale_n is None:
        scale_n = MAX_SCALE

    summary = run_accuracy_bench(
        sift.base_vecs,
        sift.query_vecs,
        sift.gt_vecs,
        top_k=top_k,
        name=name,
        n_list=n_list,
        M=M,
        K=K,
        n_probe=n_probe,
        scale_n=scale_n,
        num_runs=num_runs,
        cluster_bound=cluster_bound,
        force_recompute=force_recompute,
    )
    print(summary)
    print((time.time() - stime) / 3600)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Accuracy benchmark for SIFT/GIST: compare standard IVF-PQ vs ZK IVF-PQ "
            "against provided ground-truth (ivecs). Results are cached under data/acc_bench/."
        )
    )
    parser.add_argument(
        "--data-name",
        type=str,
        required=True,
        choices=["sift", "gist"],
        help="Dataset folder name under --data-root (expects {name}_base/query/groundtruth.*).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory that contains the dataset folders (default: data).",
    )
    parser.add_argument("--n-list", type=int, default=1024)
    parser.add_argument("--n-probe", type=int, default=8)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--cluster-bound",
        type=int,
        default=2048,
        help="ZK training cluster upperbound; set <=0 to disable.",
    )
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--scale-n",
        type=int,
        default=None,
        help="Integer rescale bound for ZK version (default: use MAX_SCALE from ivf_pq).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Cache name prefix (default: NEW_{data-name}).",
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cluster_bound = int(args.cluster_bound)
    if cluster_bound <= 0:
        cluster_bound = None

    main(
        args.data_name,
        n_list=int(args.n_list),
        n_probe=int(args.n_probe),
        cluster_bound=cluster_bound,
        data_root=str(args.data_root),
        M=int(args.M),
        K=int(args.K),
        top_k=int(args.top_k),
        num_runs=int(args.num_runs),
        scale_n=args.scale_n,
        force_recompute=bool(args.force_recompute),
        name=args.name,
    )
