from __future__ import annotations
import argparse
from typing import Callable, List, Tuple

WEIGHTED_PHASES = ("train", "finalists", "debounce", "aggregate", "probes")


def fold_selector(text: str) -> Callable[[int], List[int]]:
    # 'i/n': every n-th fold starting at i; otherwise a comma-separated list of folds.
    if "/" in text:
        i, n = (int(v) for v in text.split("/", 1))
        if not 0 <= i < n:
            raise ValueError(text)
        return lambda n_runs: list(range(i, n_runs, n))
    folds = [int(v) for v in text.split(",") if v.strip()]
    return lambda n_runs: folds


def parse(phases: Tuple[str, ...], weighted: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OHCA detection pipeline.")
    parser.add_argument("--phase", choices=phases, required=True, help="'all' runs every phase in order.")
    parser.add_argument("--folds", default="0/1",
                        help="'i/n': every n-th fold starting at i (parallel workers), or a comma-separated list.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: the least-used GPU).")
    parser.add_argument("--gpu-mem", type=int, default=None,
                        help="GPU memory cap in MB, for several workers on one GPU (default: memory growth).")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads per process, for several workers per machine (default: all cores).")
    if weighted:
        parser.add_argument("--weight", default=None, help="Gap-penalty weight, comma-separated weights, or 'all'.")
    args = parser.parse_args()
    if weighted and args.phase in WEIGHTED_PHASES and args.weight is None:
        parser.error(f"--phase {args.phase} needs --weight")
    args.weight = getattr(args, "weight", None)
    try:
        args.folds = fold_selector(args.folds)
    except ValueError:
        parser.error(f"invalid --folds value {args.folds!r}")
    return args
