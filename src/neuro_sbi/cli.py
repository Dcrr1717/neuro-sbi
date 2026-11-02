"""Command-line interface.

neuro-sbi run   --config {tiny,small,full} [--out DIR] [--workers N]
neuro-sbi infer --posterior results/posterior.pt [--set g_Na=40 ...]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import numpy as np


def _cmd_run(args: argparse.Namespace) -> int:
    from neuro_sbi.config import get_config
    from neuro_sbi.pipeline import run

    overrides = {"seed": args.seed} if args.seed is not None else {}
    cfg = get_config(args.config, **overrides)
    run(
        cfg,
        out_dir=args.out,
        workers=args.workers,
        cache_dir=None if args.no_cache else args.cache,
        reuse_posterior=args.reuse_posterior,
    )
    print((args.out or cfg.output_dir) + "/report.md")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    """Amortized inference on a new observation: one simulation, no retraining."""
    from neuro_sbi.batch import simulate
    from neuro_sbi.features import FEATURE_NAMES
    from neuro_sbi.inference import load_posterior, sample

    loaded = load_posterior(args.posterior)
    posterior, space = loaded.posterior, loaded.space
    overrides: dict[str, float] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in space.names:
            raise SystemExit(f"--set {key}: not an inferred parameter {space.names}")
        overrides[key] = float(value)
    theta = space.ground_truth(overrides)
    x, _ = simulate(space.to_physical(theta[None]), loaded.simulation, seed=args.seed)
    print("observation:", ", ".join(f"{n}={v:.3g}" for n, v in zip(FEATURE_NAMES, x[0], strict=True)))
    post = sample(posterior, x[0], args.samples)
    print(f"{'parameter':>10} {'truth':>10} {'median':>10} {'2.5%':>10} {'97.5%':>10}")
    for j, spec in enumerate(space.specs):
        q = spec.to_physical(np.quantile(post[:, j], [0.025, 0.5, 0.975]))
        tv = float(spec.to_physical(np.array(theta[j])))
        print(f"{spec.symbol:>10} {tv:10.4g} {q[1]:10.4g} {q[0]:10.4g} {q[2]:10.4g}  {spec.unit}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuro-sbi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--config", default="small", choices=["tiny", "small", "full"])
    p_run.add_argument("--out", default=None, help="output directory (default: per config)")
    p_run.add_argument("--workers", type=int, default=None, help="simulation processes")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--cache", default=".cache", help="training-set cache directory")
    p_run.add_argument("--no-cache", action="store_true")
    p_run.add_argument(
        "--reuse-posterior", action="store_true", help="load OUT/posterior.pt instead of retraining"
    )
    p_run.set_defaults(func=_cmd_run)

    p_inf = sub.add_parser("infer", help="infer parameters of a new simulated cell")
    p_inf.add_argument("--posterior", default="results/posterior.pt")
    p_inf.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="physical value of an inferred parameter (default: reference cell)",
    )
    p_inf.add_argument("--samples", type=int, default=10_000)
    p_inf.add_argument("--seed", type=int, default=0)
    p_inf.set_defaults(func=_cmd_infer)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    return int(args.func(args))
