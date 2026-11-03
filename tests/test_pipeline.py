"""End-to-end tests on the ``tiny`` configuration (offline, ~20 s)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neuro_sbi.cli import main
from neuro_sbi.config import CONFIGS, TINY, get_config
from neuro_sbi.features import N_FEATURES
from neuro_sbi.inference import load_posterior, log_prob_unnormalized, save_posterior, train_ensemble
from neuro_sbi.params import ParameterSpace
from neuro_sbi.pipeline import _joint_at_report_levels, _marginal_at_report_levels, derive_seed, run

FIGURES = (
    "posterior_pairplot.png",
    "ppc_traces.png",
    "ppc_features.png",
    "sbc_ranks.png",
    "sbc_ecdf.png",
    "coverage.png",
    "recovery.png",
    "training_curve.png",
)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("tiny")
    return out, run(TINY, out_dir=out, workers=1, cache_dir=out / "cache")


def test_pipeline_writes_all_outputs(tiny_run: tuple[Path, dict[str, Any]]) -> None:
    out, metrics = tiny_run
    for name in FIGURES:
        assert (out / "figures" / name).stat().st_size > 10_000, name
    for name in ("metrics.json", "report.md", "posterior.pt", "training.json", "sbc.npz"):
        assert (out / name).exists(), name
    assert json.loads((out / "metrics.json").read_text())["config"]["name"] == "tiny"
    assert "Simulation-based calibration" in (out / "report.md").read_text()
    assert metrics["training_data"]["all_finite"]


def test_pipeline_metrics_are_well_formed(tiny_run: tuple[Path, dict[str, Any]]) -> None:
    _, m = tiny_run
    names = m["parameters"]
    assert set(m["sbc"]["per_parameter"]) == set(names)
    for t in m["sbc"]["per_parameter"].values():
        assert 0.0 <= t["ks_pvalue"] <= 1.0 and 0.0 <= t["chi2_pvalue"] <= 1.0
    for level, cov in m["sbc"]["marginal_coverage"].items():
        assert all(0.0 <= c <= 1.0 for c in cov.values()), level
    wr = m["worked_recovery"]
    assert len(wr["marginals"]) == len(names)
    assert 0.0 <= wr["hpd_level_of_truth"] <= 1.0
    assert len(m["ppc"]["posterior"]) == N_FEATURES
    groups = m["recovery_by_group"]
    assert sum(g["num"] for g in groups.values()) == m["sbc"]["num_sbc"]
    for g in groups.values():
        assert set(g["per_parameter"]) == set(names)
    k = TINY.training.ensemble_size
    assert m["training"]["ensemble_size"] == len(m["training"]["members"]) == k
    assert len({mem["seed"] for mem in m["training"]["members"]}) == k
    assert len(m["sbc"]["members"]) == k
    for mem in m["sbc"]["members"]:
        assert set(mem["marginal_coverage_95"]) == set(names)
        assert 0 <= mem["num_uniform_at_bonferroni"] <= len(names)
    hpd_groups = m["sbc"]["joint_hpd_coverage_by_group"]
    assert sum(g["num"] for g in hpd_groups.values()) == m["sbc"]["num_sbc"]
    for g in hpd_groups.values():
        assert all(0.0 <= g[level] <= 1.0 for level in ("0.50", "0.80", "0.90", "0.95"))


def test_sbc_arrays_are_saved(tiny_run: tuple[Path, dict[str, Any]]) -> None:
    out, m = tiny_run
    data = np.load(out / "sbc.npz")
    n, dim = m["sbc"]["num_sbc"], len(m["parameters"])
    assert data["theta"].shape == data["ranks"].shape == data["posterior_mean"].shape == (n, dim)
    assert data["x"].shape == (n, N_FEATURES)
    assert np.all((data["hpd_level"] >= 0) & (data["hpd_level"] <= 1))
    assert np.all((data["ranks"] >= 0) & (data["ranks"] <= m["sbc"]["posterior_samples"]))
    k = TINY.training.ensemble_size
    assert data["member_ranks"].shape == (k, n, dim) and data["member_hpd_level"].shape == (k, n)


def test_reported_coverage_is_exact_at_each_level(tiny_run: tuple[Path, dict[str, Any]]) -> None:
    # Regression: coverage at 95 % was once read off a 0.02-spaced grid of
    # levels, i.e. reported the 94 % (or 96 %) level instead.
    out, m = tiny_run
    data = np.load(out / "sbc.npz")
    sbc = m["sbc"]
    for g in (0.5, 0.8, 0.9, 0.95):
        key = f"{g:.2f}"
        assert sbc["joint_hpd_coverage"][key] == pytest.approx(np.mean(data["hpd_level"] < g))
        u = data["ranks"] / sbc["posterior_samples"]
        expected = np.mean((u >= (1 - g) / 2) & (u <= (1 + g) / 2), axis=0)
        np.testing.assert_allclose(list(sbc["marginal_coverage"][key].values()), expected)
    # A truth at HPD level 0.945 lies inside the 95 % region but outside 94 %.
    assert _joint_at_report_levels(np.array([0.945]))["0.95"] == 1.0
    ranks = np.array([[26]])  # u = 0.026: inside the central 95 % interval, outside 94 %
    assert _marginal_at_report_levels(ranks, 1000, ("a",))["0.95"]["a"] == 1.0
    for k, member in enumerate(sbc["members"]):
        assert member["joint_hpd_coverage"]["0.95"] == pytest.approx(
            np.mean(data["member_hpd_level"][k] < 0.95)
        )


def test_reuse_posterior_reproduces_metrics(tiny_run: tuple[Path, dict[str, Any]]) -> None:
    out, m = tiny_run
    m2 = run(TINY, out_dir=out, workers=1, cache_dir=out / "cache", reuse_posterior=True)
    assert m2["worked_recovery"]["marginals"] == m["worked_recovery"]["marginals"]
    assert m2["sbc"]["per_parameter"] == m["sbc"]["per_parameter"]


def test_infer_cli(tiny_run: tuple[Path, dict[str, Any]], capsys: pytest.CaptureFixture[str]) -> None:
    out, _ = tiny_run
    assert (
        main(["infer", "--posterior", str(out / "posterior.pt"), "--set", "g_Na=30", "--samples", "200"]) == 0
    )
    text = capsys.readouterr().out
    assert "gNa" in text and "VT" in text


def test_saved_posterior_roundtrip(tmp_path: Path) -> None:
    space = ParameterSpace()
    rng = np.random.default_rng(0)
    theta = space.sample_prior(256, rng)
    x = np.abs(rng.standard_normal((256, N_FEATURES))) + theta[:, :1]
    cfg = dataclasses.replace(TINY.training, max_num_epochs=2, ensemble_size=2)
    posterior, _ = train_ensemble(theta, x, space, cfg, seeds=(0, 1))
    save_posterior(posterior, tmp_path / "p.pt", space, cfg, TINY.simulation)
    loaded = load_posterior(tmp_path / "p.pt")
    assert loaded.space.names == space.names and loaded.simulation == TINY.simulation
    assert loaded.training == cfg
    th = theta[None, :8]
    np.testing.assert_allclose(
        log_prob_unnormalized(posterior, th, x[:8]),
        log_prob_unnormalized(loaded.posterior, th, x[:8]),
        rtol=1e-5,
    )
    # The loaded posterior accepts raw (unstandardized) features directly.
    s = loaded.posterior.sample((10,), x=torch.as_tensor(x[0], dtype=torch.float32), show_progress_bars=False)
    assert s.shape == (10, space.dim)


def test_configs_and_seeds() -> None:
    assert set(CONFIGS) == {"tiny", "small", "full"}
    assert get_config("small", seed=1).seed == 1
    with pytest.raises(KeyError):
        get_config("huge")
    seeds = {derive_seed(0, s) for s in ("train-theta", "train-sim", "sbc-theta", "sbc-sim", "observation")}
    assert len(seeds) == 5
    assert derive_seed(0, "x") == derive_seed(0, "x") != derive_seed(1, "x")
