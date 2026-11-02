"""Validate the diagnostics on a conjugate Gaussian model with an exact posterior.

theta ~ N(0, 1), x | theta ~ N(theta, s^2)  =>  theta | x ~ N(x / (1 + s^2), s^2 / (1 + s^2)).
"""

from __future__ import annotations

import numpy as np
import pytest

from neuro_sbi import diagnostics as dg

N, L, D, S2 = 2000, 200, 2, 0.5


def _posterior_draws(scale: float, shift: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((N, D))
    x = theta + np.sqrt(S2) * rng.standard_normal((N, D))
    mean = x / (1 + S2) + shift
    sd = np.sqrt(S2 / (1 + S2)) * scale
    samples = mean[None] + sd * rng.standard_normal((L, N, D))
    return samples, theta


def test_rank_bin_probabilities_are_exact() -> None:
    p = dg.rank_bin_probabilities(9, 5)
    np.testing.assert_allclose(p, [0.2] * 5)
    p = dg.rank_bin_probabilities(10, 3)  # 11 ranks into 3 bins: 4, 4, 3
    np.testing.assert_allclose(p, np.array([4, 4, 3]) / 11)


def test_exact_posterior_is_calibrated() -> None:
    samples, theta = _posterior_draws(scale=1.0)
    ranks = dg.sbc_ranks(samples, theta)
    assert np.min(ranks) >= 0 and np.max(ranks) <= L
    for t in dg.rank_uniformity_tests(ranks, L, 20, seed=0):
        assert t["ks_pvalue"] > 1e-3 and t["chi2_pvalue"] > 1e-3
    levels = np.array([0.5, 0.9])
    cov = dg.central_coverage(ranks, L, levels)
    np.testing.assert_allclose(cov, levels[:, None].repeat(D, axis=1), atol=0.04)
    rec = dg.recovery_metrics(samples, theta, np.array([[-3.0, 3.0]] * D))
    np.testing.assert_allclose(rec["z_mean"], 0.0, atol=0.1)
    np.testing.assert_allclose(rec["z_std"], 1.0, atol=0.1)


@pytest.mark.parametrize(("scale", "direction"), [(0.5, -1), (2.0, +1)])
def test_miscalibration_is_detected(scale: float, direction: int) -> None:
    samples, theta = _posterior_draws(scale=scale)
    ranks = dg.sbc_ranks(samples, theta)
    assert all(t["ks_pvalue"] < 1e-6 for t in dg.rank_uniformity_tests(ranks, L, 20))
    cov = dg.central_coverage(ranks, L, np.array([0.5]))[0]
    assert np.all(np.sign(cov - 0.5) == direction)  # over-confident -> under-covers


def test_bias_is_detected() -> None:
    samples, theta = _posterior_draws(scale=1.0, shift=0.5)
    ranks = dg.sbc_ranks(samples, theta)
    assert np.mean(ranks) < 0.4 * L  # truth sits low in a right-shifted posterior


def test_hpd_coverage_of_exact_gaussian() -> None:
    rng = np.random.default_rng(3)
    theta = rng.standard_normal((N, D))
    x = theta + np.sqrt(S2) * rng.standard_normal((N, D))
    mean, var = x / (1 + S2), S2 / (1 + S2)
    samples = mean[None] + np.sqrt(var) * rng.standard_normal((L, N, D))

    def logq(t: np.ndarray) -> np.ndarray:
        return np.asarray(-0.5 * np.sum((t - mean) ** 2, axis=-1) / var)

    alpha = dg.hpd_levels(logq(samples), logq(theta[None])[0])
    levels = np.array([0.25, 0.5, 0.75, 0.95])
    np.testing.assert_allclose(dg.hpd_coverage(alpha, levels), levels, atol=0.04)


def test_bands_and_predictive_check() -> None:
    lo, hi = dg.histogram_band(1000, 20, alpha=0.01)
    assert lo < 50 < hi
    assert 0.0 < dg.dkw_epsilon(1000) < 0.05
    clo, chi = dg.coverage_band(1000, np.array([0.5]))
    assert clo[0] < 0.5 < chi[0]
    pred = np.array([[0.0], [1.0], [1.0], [2.0]])
    res = dg.predictive_check(np.array([1.0]), pred)[0]
    assert res["quantile"] == pytest.approx(0.5)  # ties counted half
    assert res["p_value"] == pytest.approx(1.0)


def test_marginal_summary() -> None:
    rng = np.random.default_rng(0)
    s = rng.standard_normal((20000, 1))
    out = dg.marginal_summary(s, np.array([0.0]))[0]
    assert out["inside_ci95"]
    assert out["percentile"] == pytest.approx(50.0, abs=1.5)
    assert not dg.marginal_summary(s, np.array([3.0]))[0]["inside_ci95"]
