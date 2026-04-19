import argparse
import time
from pathlib import Path

from vec_data_load.sift import SIFT
from bench.acc_bench import DEFAULT_REPORT_KS, run_accuracy_bench
from ivf_pq import MAX_SCALE


def _parse_report_ks(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    raw = str(arg).strip()
    if not raw:
        return None
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def _resolve_layout(data_name: str, layout_arg: str | None) -> str | None:
    token = "auto" if layout_arg is None else str(layout_arg).strip().lower()
    if token in ("", "auto"):
        if data_name in {"sift", "gist"}:
            return "mod8"
        return None
    if token == "none":
        return None
    if token == "mod8":
        return token
    raise ValueError(f"Unsupported layout: {layout_arg!r}")


def _print_metric_block(summary: dict, *, group: str, report_ks: list[int]) -> None:
    block = summary.get(group, {})
    if not isinstance(block, dict):
        return

    print(f"[acc_bench] {group}")
    for k in report_ks:
        pass_key = f"mean_pass_at_{k}"
        recall_key = f"mean_recall_at_{k}"
        if pass_key not in block and recall_key not in block:
            continue

        pass_val = block.get(pass_key)
        recall_val = block.get(recall_key)
        pass_ci = block.get(f"ci95_pass_at_{k}")
        recall_ci = block.get(f"ci95_recall_at_{k}")

        pass_txt = (
            f"{float(pass_val):.4f}±{float(pass_ci):.4f}"
            if pass_val is not None and pass_ci is not None
            else "n/a"
        )
        recall_txt = (
            f"{float(recall_val):.4f}±{float(recall_ci):.4f}"
            if recall_val is not None and recall_ci is not None
            else "n/a"
        )
        print(f"  @ {k:>3d}: pass={pass_txt}  recall={recall_txt}")

    for metric in ("train_time", "query_time", "changed_count"):
        mean_key = f"mean_{metric}"
        if mean_key in block:
            ci_key = f"ci95_{metric}"
            value = float(block[mean_key])
            ci = float(block.get(ci_key, 0.0))
            print(f"  {metric}: {value:.4f}±{ci:.4f}")


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
    report_ks: list[int] | None = None,
    layout: str | None = "auto",
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

    resolved_layout = _resolve_layout(data_name, layout)
    print(f"[acc_bench] layout={resolved_layout or 'none'}")

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
        report_ks=report_ks,
        layout=resolved_layout,
    )
    print(summary)
    raw_ks = report_ks if report_ks is not None else list(DEFAULT_REPORT_KS)
    resolved_report_ks = sorted({int(k) for k in raw_ks} | {int(top_k)})

    _print_metric_block(summary, group="standard", report_ks=resolved_report_ks)
    _print_metric_block(summary, group="zk", report_ks=resolved_report_ks)
    print((time.time() - stime) / 3600)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Accuracy benchmark for SIFT/GIST: compare standard IVF-PQ vs ZK IVF-PQ "
            "against provided ground-truth (ivecs), and report both pass@k and recall@k "
            "for multiple ks from one run. Results are cached under data/acc_bench/."
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
        "--report-ks",
        type=str,
        default="",
        help=(
            "Comma-separated ks to report from the single top-k retrieval "
            f"(default: {','.join(str(k) for k in DEFAULT_REPORT_KS)}; top-k is always included)."
        ),
    )
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
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "none", "mod8"],
        help=(
            "Vector layout before PQ. 'auto' defaults to mod8 for sift/gist, "
            "'none' keeps the raw dimension order."
        ),
    )
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
        report_ks=_parse_report_ks(args.report_ks),
        layout=str(args.layout),
    )
