"""End-to-end pipeline: simulate -> train NPE -> worked recovery -> PPC -> SBC.

Every stage uses its own random stream derived from ``config.seed``, so the
SBC/test simulations are independent of the training simulations and the
observation used in the worked example is never seen during training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from neuro_sbi import diagnostics as dg
from neuro_sbi import plotting
from neuro_sbi.batch import default_workers, simulate
from neuro_sbi.config import PipelineConfig
from neuro_sbi.features import FEATURE_NAMES, FEATURE_UNITS
from neuro_sbi.inference import (
    ensemble_members,
    load_posterior,
    log_prob_unnormalized,
    sample,
    sample_batched,
    save_posterior,
    train_ensemble,
)
from neuro_sbi.params import ParameterSpace
from neuro_sbi.report import write_report

FloatArray = NDArray[np.float64]
log = logging.getLogger("neuro_sbi")

# Credibility levels: a fine grid for the coverage plots, and the levels whose
# coverage is reported (computed exactly, not read off the grid).
COVERAGE_LEVELS = np.linspace(0.0, 1.0, 101)
REPORT_LEVELS = (0.5, 0.8, 0.9, 0.95)


def derive_seed(seed: int, stream: str) -> int:
    """Independent, reproducible integer seed for a named random stream."""
    ss = np.random.SeedSequence([seed, zlib.crc32(stream.encode())])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def member_seed(seed: int, k: int) -> int:
    """Training seed of ensemble member ``k``; member 0 uses the stream a
    single-flow run has always used."""
    return derive_seed(seed, "torch" if k == 0 else f"torch-{k}")


def _cache_key(cfg: PipelineConfig) -> str:
    blob = json.dumps(
        {
            "simulation": cfg.to_dict()["simulation"],
            "inferred": cfg.inferred,
            "n": cfg.num_simulations,
            "seed": cfg.seed,
            "version": 1,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def simulate_training_set(
    cfg: PipelineConfig, space: ParameterSpace, workers: int, cache_dir: Path | None
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Draw ``theta ~ prior`` and simulate summary statistics (cached on disk).

    Returns ``(theta, x, info)``; ``info`` records whether the set came from
    the cache and how long the original simulation took.
    """
    path = cache_dir / f"train_{cfg.name}_{_cache_key(cfg)}.npz" if cache_dir else None
    if path is not None and path.exists():
        data = np.load(path)
        log.info("loaded cached training set %s", path)
        info: dict[str, Any] = {"cached": True}
        if "seconds" in data:
            info |= {"seconds": float(data["seconds"]), "workers": int(data["workers"])}
        return data["theta"], data["x"], info
    t0 = time.perf_counter()
    rng = np.random.default_rng(derive_seed(cfg.seed, "train-theta"))
    theta = space.sample_prior(cfg.num_simulations, rng)
    x, _ = simulate(
        space.to_physical(theta),
        cfg.simulation,
        seed=derive_seed(cfg.seed, "train-sim"),
        chunk_size=cfg.chunk_size,
        workers=workers,
    )
    seconds = time.perf_counter() - t0
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, theta=theta, x=x, seconds=seconds, workers=workers)
    return theta, x, {"cached": False, "seconds": seconds, "workers": workers}


def run(
    cfg: PipelineConfig,
    out_dir: str | Path | None = None,
    workers: int | None = None,
    cache_dir: str | Path | None = ".cache",
    reuse_posterior: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline and write figures, metrics and a report.

    Args:
        cfg: Pipeline configuration.
        out_dir: Output directory (default ``cfg.output_dir``).
        workers: Simulation worker processes (default: all CPUs).
        cache_dir: Where to cache the training set (``None`` disables caching).
        reuse_posterior: Load ``out_dir/posterior.pt`` instead of retraining
            (validation stages are re-run from scratch).

    Returns the metrics dictionary (also written to ``metrics.json``).
    """
    out = Path(out_dir if out_dir is not None else cfg.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or default_workers()
    torch.set_num_threads(max(1, workers))
    cache = Path(cache_dir) if cache_dir else None
    space = ParameterSpace(cfg.inferred)
    settings = cfg.simulation
    dcfg = cfg.diagnostics
    bounds = space.bounds
    labels = space.labels
    timings: dict[str, float] = {}
    metrics: dict[str, Any] = {
        "config": cfg.to_dict(),
        "parameters": list(space.names),
        "parameter_labels": labels,
        "features": list(FEATURE_NAMES),
        "workers": workers,
        "versions": _versions(),
    }

    # 1. Training data ---------------------------------------------------------
    t0 = time.perf_counter()
    theta, x, sim_info = simulate_training_set(cfg, space, workers, cache)
    timings["simulate_training"] = float(sim_info.get("seconds", time.perf_counter() - t0))
    metrics["training_data"] = {
        **sim_info,
        "num_simulations": int(theta.shape[0]),
        "fraction_spiking": float(np.mean(x[:, 0] > 0)),
        "all_finite": bool(np.all(np.isfinite(x))),
    }
    log.info("training set: %d simulations (%.1fs)", theta.shape[0], timings["simulate_training"])

    # 2. Train the amortized posterior ----------------------------------------
    t0 = time.perf_counter()
    post_path, info_path = out / "posterior.pt", out / "training.json"
    if reuse_posterior and post_path.exists() and info_path.exists():
        loaded = load_posterior(post_path)
        if loaded.space.names != space.names or loaded.simulation != settings:
            raise ValueError(f"{post_path} was trained with a different configuration")
        posterior = loaded.posterior
        info = json.loads(info_path.read_text())
        log.info("reusing trained posterior %s", post_path)
    else:
        seeds = [member_seed(cfg.seed, k) for k in range(cfg.training.ensemble_size)]
        posterior, member_info = train_ensemble(theta, x, space, cfg.training, seeds, workers=workers)
        info = {"members": member_info, "train_seconds": time.perf_counter() - t0, "workers": workers}
        save_posterior(posterior, post_path, space, cfg.training, settings)
        info_path.write_text(json.dumps(info))
    members_info: list[dict[str, Any]] = info["members"]
    timings["train"] = float(info["train_seconds"])
    metrics["training"] = {
        "ensemble_size": len(members_info),
        "num_parameters": sum(m["num_parameters"] for m in members_info),
        "learning_rates": members_info[0]["learning_rates"],
        "train_seconds": timings["train"],
        "members": [
            {
                "seed": m["seed"],
                "num_parameters": m["num_parameters"],
                "epochs_trained": m["epochs_trained"],
                "phase_end_epochs": m["phase_end_epochs"],
                "best_validation_log_prob": -m["best_validation_loss"],
                "final_training_log_prob": -m["training_loss"][-1],
                "final_validation_log_prob": -m["validation_loss"][-1],
            }
            for m in members_info
        ],
    }
    plotting.training_curve(
        [m["training_loss"] for m in members_info],
        [m["validation_loss"] for m in members_info],
        fig_dir / "training_curve.png",
        phase_ends=[m["phase_end_epochs"] for m in members_info],
        learning_rates=members_info[0]["learning_rates"],
    )
    log.info(
        "trained %d flow(s) for %s epochs in %.1fs",
        len(members_info),
        "/".join(str(m["epochs_trained"]) for m in members_info),
        timings["train"],
    )

    # 3. Worked recovery on a held-out observation ------------------------------
    t0 = time.perf_counter()
    theta_gt = space.ground_truth()
    x_o, v_o = simulate(
        space.to_physical(theta_gt[None]),
        settings,
        seed=derive_seed(cfg.seed, "observation"),
        keep_traces=True,
    )
    assert v_o is not None
    torch.manual_seed(derive_seed(cfg.seed, "sample-observation"))
    post = sample(posterior, x_o[0], dcfg.num_posterior_samples)
    lq_s = log_prob_unnormalized(posterior, post[:, None, :], x_o)[:, 0]
    lq_t = log_prob_unnormalized(posterior, theta_gt[None, None, :], x_o)[0, 0]
    hpd_level = float(np.mean(lq_s > lq_t))
    summary = dg.marginal_summary(post, theta_gt)
    for j, spec in enumerate(space.specs):
        q = np.quantile(post[:, j], [0.025, 0.5, 0.975])
        phys = spec.to_physical(q)
        summary[j].update(
            {
                "name": spec.name,
                "label": spec.axis_label,
                "unit": spec.unit,
                "log_scale": spec.log_scale,
                "truth_physical": spec.default,
                "median_physical": float(phys[1]),
                "ci95_physical": [float(phys[0]), float(phys[2])],
            }
        )
    metrics["worked_recovery"] = {
        "observation": dict(zip(FEATURE_NAMES, map(float, x_o[0]), strict=True)),
        "marginals": summary,
        "hpd_level_of_truth": hpd_level,
        "all_inside_ci95": all(s["inside_ci95"] for s in summary),
    }
    plotting.pairplot(
        post,
        theta_gt,
        bounds,
        labels,
        fig_dir / "posterior_pairplot.png",
        title=f"NPE posterior for one held-out observation ({cfg.num_simulations:,} training simulations)",
    )
    timings["worked_recovery"] = time.perf_counter() - t0

    # 4. Posterior predictive check -----------------------------------------
    t0 = time.perf_counter()
    n_ppc = min(dcfg.num_ppc, post.shape[0])
    x_post, v_post = simulate(
        space.to_physical(post[:n_ppc]),
        settings,
        seed=derive_seed(cfg.seed, "ppc"),
        chunk_size=cfg.chunk_size,
        workers=workers,
        keep_traces=True,
    )
    theta_pr = space.sample_prior(n_ppc, np.random.default_rng(derive_seed(cfg.seed, "prior-pred")))
    x_prior, v_prior = simulate(
        space.to_physical(theta_pr),
        settings,
        seed=derive_seed(cfg.seed, "prior-pred-sim"),
        chunk_size=cfg.chunk_size,
        workers=workers,
        keep_traces=True,
    )
    assert v_post is not None and v_prior is not None
    ppc_post = dg.predictive_check(x_o[0], x_post)
    ppc_prior = dg.predictive_check(x_o[0], x_prior)
    metrics["ppc"] = {
        "num_samples": n_ppc,
        "posterior": dict(zip(FEATURE_NAMES, ppc_post, strict=True)),
        "prior": dict(zip(FEATURE_NAMES, ppc_prior, strict=True)),
        "posterior_spike_count_exact_match": float(np.mean(x_post[:, 0] == x_o[0, 0])),
        "prior_spike_count_exact_match": float(np.mean(x_prior[:, 0] == x_o[0, 0])),
    }
    k = dcfg.num_ppc_traces
    plotting.ppc_traces(
        settings.time(),
        v_o[0],
        v_post[:k],
        v_prior[:k],
        settings.stimulus(),
        settings.t_on,
        settings.t_off,
        fig_dir / "ppc_traces.png",
    )
    plotting.ppc_features(
        x_o[0],
        x_post,
        x_prior,
        FEATURE_NAMES,
        [FEATURE_UNITS[f] for f in FEATURE_NAMES],
        [p["quantile"] for p in ppc_post],
        fig_dir / "ppc_features.png",
    )
    del v_post, v_prior
    timings["ppc"] = time.perf_counter() - t0

    # 5. Simulation-based calibration on independent simulations --------------
    t0 = time.perf_counter()
    theta_sbc = space.sample_prior(dcfg.num_sbc, np.random.default_rng(derive_seed(cfg.seed, "sbc-theta")))
    x_sbc, _ = simulate(
        space.to_physical(theta_sbc),
        settings,
        seed=derive_seed(cfg.seed, "sbc-sim"),
        chunk_size=cfg.chunk_size,
        workers=workers,
    )
    torch.manual_seed(derive_seed(cfg.seed, "sample-sbc"))
    samples_sbc = sample_batched(posterior, x_sbc, dcfg.sbc_samples)
    ranks = dg.sbc_ranks(samples_sbc, theta_sbc)
    tests = dg.rank_uniformity_tests(ranks, dcfg.sbc_samples, dcfg.sbc_bins, seed=cfg.seed)
    marg_cov = dg.central_coverage(ranks, dcfg.sbc_samples, COVERAGE_LEVELS)
    lq_samples = log_prob_unnormalized(posterior, samples_sbc, x_sbc)
    lq_truth = log_prob_unnormalized(posterior, theta_sbc[None], x_sbc)[0]
    alpha = dg.hpd_levels(lq_samples, lq_truth)
    joint_cov = dg.hpd_coverage(alpha, COVERAGE_LEVELS)
    ks_alpha = float(stats.kstest(alpha, "uniform").pvalue)
    rec = dg.recovery_metrics(samples_sbc, theta_sbc, bounds)
    spiking = x_sbc[:, 0] > 0
    bonferroni = 0.05 / space.dim
    # Every member of an ensemble on its own, on the same simulations: shows
    # what the mixture changes.
    members = ensemble_members(posterior)
    member_stats: list[dict[str, Any]] = []
    member_joint_cov: list[FloatArray] = []
    member_alpha: list[FloatArray] = []
    member_ranks: list[NDArray[np.int64]] = []
    for k, member in enumerate(members if len(members) > 1 else []):
        torch.manual_seed(derive_seed(cfg.seed, f"sample-sbc-member-{k}"))
        s_k = sample_batched(member, x_sbc, dcfg.sbc_samples)
        r_k = dg.sbc_ranks(s_k, theta_sbc)
        lq_truth_k = log_prob_unnormalized(member, theta_sbc[None], x_sbc)[0]
        a_k = dg.hpd_levels(log_prob_unnormalized(member, s_k, x_sbc), lq_truth_k)
        del s_k
        jc_k = dg.hpd_coverage(a_k, COVERAGE_LEVELS)
        tests_k = dg.rank_uniformity_tests(r_k, dcfg.sbc_samples, dcfg.sbc_bins, seed=cfg.seed)
        member_stats.append(
            {
                "heldout_log_prob": float(np.mean(lq_truth_k)),
                "joint_hpd_coverage": _joint_at_report_levels(a_k),
                "joint_hpd_ks_pvalue": float(stats.kstest(a_k, "uniform").pvalue),
                "marginal_coverage_95": _marginal_at_report_levels(r_k, dcfg.sbc_samples, space.names)[
                    "0.95"
                ],
                "num_uniform_at_bonferroni": int(sum(t["ks_pvalue"] > bonferroni for t in tests_k)),
            }
        )
        member_joint_cov.append(jc_k)
        member_alpha.append(a_k)
        member_ranks.append(r_k)
    rec_by_group = {
        group: dg.recovery_metrics(samples_sbc[:, sel], theta_sbc[sel], bounds)
        for group, sel in (("spiking", spiking), ("silent", ~spiking))
        if int(sel.sum()) >= 2
    }
    timings["sbc"] = time.perf_counter() - t0

    metrics["sbc"] = {
        "num_sbc": dcfg.num_sbc,
        "posterior_samples": dcfg.sbc_samples,
        "bins": dcfg.sbc_bins,
        "per_parameter": {
            name: tests[j] | {"uniform_at_bonferroni_0.05": bool(tests[j]["ks_pvalue"] > bonferroni)}
            for j, name in enumerate(space.names)
        },
        "bonferroni_threshold": bonferroni,
        "marginal_coverage": _marginal_at_report_levels(ranks, dcfg.sbc_samples, space.names),
        "joint_hpd_coverage": _joint_at_report_levels(alpha),
        "joint_hpd_ks_pvalue": ks_alpha,
        "heldout_log_prob": float(np.mean(lq_truth)),
        "members": member_stats,
        "joint_hpd_coverage_by_group": {
            group: {"num": int(sel.sum())} | _joint_at_report_levels(alpha[sel])
            for group, sel in (("spiking", spiking), ("silent", ~spiking))
            if int(sel.sum()) >= 2
        },
    }
    metrics["recovery"] = {
        name: {key: float(val[j]) for key, val in rec.items()} for j, name in enumerate(space.names)
    }
    metrics["recovery_by_group"] = {
        group: {
            "num": int((spiking if group == "spiking" else ~spiking).sum()),
            "per_parameter": {
                name: {key: float(val[j]) for key, val in r.items()} for j, name in enumerate(space.names)
            },
        }
        for group, r in rec_by_group.items()
    }
    plotting.sbc_rank_histograms(
        ranks,
        dcfg.sbc_samples,
        dcfg.sbc_bins,
        labels,
        [t["ks_pvalue"] for t in tests],
        fig_dir / "sbc_ranks.png",
    )
    plotting.sbc_ecdf(ranks, dcfg.sbc_samples, labels, fig_dir / "sbc_ecdf.png")
    plotting.coverage_plot(
        COVERAGE_LEVELS,
        marg_cov,
        joint_cov,
        dcfg.num_sbc,
        labels,
        fig_dir / "coverage.png",
        member_joint=member_joint_cov,
    )
    plotting.recovery_plot(
        theta_sbc,
        samples_sbc,
        bounds,
        labels,
        rec["contraction"],
        rec["r"],
        fig_dir / "recovery.png",
        spiking=spiking,
    )
    np.savez_compressed(
        out / "sbc.npz",
        theta=theta_sbc,
        x=x_sbc,
        ranks=ranks,
        hpd_level=alpha,
        posterior_mean=samples_sbc.mean(axis=0),
        posterior_sd=samples_sbc.std(axis=0),
        # Per member of an ensemble (empty for a single flow).
        member_ranks=np.asarray(member_ranks, dtype=np.int64).reshape(-1, *ranks.shape),
        member_hpd_level=np.asarray(member_alpha, dtype=np.float64).reshape(-1, alpha.size),
    )
    log.info("SBC on %d simulations done in %.1fs", dcfg.num_sbc, timings["sbc"])

    metrics["timings_sec"] = timings
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics, out / "report.md")
    return metrics


def _marginal_at_report_levels(
    ranks: NDArray[np.int64], num_samples: int, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    cov = dg.central_coverage(ranks, num_samples, np.asarray(REPORT_LEVELS))
    return {
        f"{g:.2f}": dict(zip(names, map(float, row), strict=True))
        for g, row in zip(REPORT_LEVELS, cov, strict=True)
    }


def _joint_at_report_levels(alpha: FloatArray) -> dict[str, float]:
    cov = dg.hpd_coverage(alpha, np.asarray(REPORT_LEVELS))
    return {f"{g:.2f}": float(c) for g, c in zip(REPORT_LEVELS, cov, strict=True)}


def _versions() -> dict[str, str]:
    import platform

    import matplotlib
    import sbi

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sbi": sbi.__version__,
        "matplotlib": matplotlib.__version__,
    }
