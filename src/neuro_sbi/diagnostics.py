"""Validation of an amortized posterior estimator.

All functions are pure numpy/scipy so they can be unit-tested against
problems with analytically known posteriors.

Simulation-based calibration (SBC; Cook et al. 2006, Talts et al. 2018)
----------------------------------------------------------------------
If theta* ~ p(theta), x ~ p(x | theta*) and theta_1..L ~ q(theta | x), then for
a *correct* posterior q = p the rank of theta*_j among the L draws of
coordinate j is uniform on {0, ..., L}. Deviations are diagnostic:
a U-shape means the estimator is over-confident, a dome means it is
under-confident (too wide), a slope means it is biased.

Expected coverage (Hermans et al. 2022)
---------------------------------------
For credibility level gamma, a calibrated posterior's gamma credible region
contains the true parameter in a fraction gamma of trials. We check this both
for central marginal intervals and for the joint highest-posterior-density
(HPD) region (using the flow's log density), which also tests the dependence
structure between parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def sbc_ranks(samples: FloatArray, theta: FloatArray) -> IntArray:
    """Rank of each true parameter among its posterior draws.

    Args:
        samples: Posterior draws, shape ``(L, n, dim)``.
        theta: True parameters, shape ``(n, dim)``.

    Returns:
        Integer ranks in ``{0, ..., L}``, shape ``(n, dim)``.
    """
    if samples.ndim != 3 or samples.shape[1:] != theta.shape:
        raise ValueError(f"shape mismatch: samples {samples.shape}, theta {theta.shape}")
    return np.sum(samples < theta[None, :, :], axis=0).astype(np.int64)


def rank_bin_probabilities(num_samples: int, n_bins: int) -> FloatArray:
    """Exact probability of each histogram bin under uniform ranks on
    ``{0, ..., L}`` when ranks are binned with ``floor(r * n_bins / (L + 1))``."""
    r = np.arange(num_samples + 1)
    bins = (r * n_bins) // (num_samples + 1)
    return np.bincount(bins, minlength=n_bins).astype(np.float64) / (num_samples + 1)


def bin_ranks(ranks: IntArray, num_samples: int, n_bins: int) -> IntArray:
    """Histogram counts, shape ``(n_bins, dim)``."""
    b = (ranks * n_bins) // (num_samples + 1)
    return np.stack([np.bincount(b[:, j], minlength=n_bins) for j in range(ranks.shape[1])], axis=1).astype(
        np.int64
    )


def rank_uniformity_tests(
    ranks: IntArray, num_samples: int, n_bins: int, seed: int = 0
) -> list[dict[str, float]]:
    """Per-dimension tests of rank uniformity.

    * chi-squared goodness of fit of the binned rank histogram;
    * Kolmogorov-Smirnov test of the randomized probability integral transform
      ``u = (r + U(0, 1)) / (L + 1)``, which is exactly U(0, 1) under H0.
    """
    rng = np.random.default_rng(seed)
    n, dim = ranks.shape
    expected = rank_bin_probabilities(num_samples, n_bins) * n
    counts = bin_ranks(ranks, num_samples, n_bins)
    out: list[dict[str, float]] = []
    for j in range(dim):
        chi2 = stats.chisquare(counts[:, j], expected)
        u = (ranks[:, j] + rng.uniform(size=n)) / (num_samples + 1)
        ks = stats.kstest(u, "uniform")
        out.append(
            {
                "chi2_stat": float(chi2.statistic),
                "chi2_pvalue": float(chi2.pvalue),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return out


def histogram_band(n: int, n_bins: int, alpha: float = 0.01) -> tuple[float, float]:
    """Pointwise (1 - alpha) band for the count in one rank-histogram bin."""
    p = 1.0 / n_bins
    return float(stats.binom.ppf(alpha / 2, n, p)), float(stats.binom.ppf(1 - alpha / 2, n, p))


def dkw_epsilon(n: int, alpha: float = 0.05) -> float:
    """Half-width of the simultaneous (1 - alpha) Dvoretzky-Kiefer-Wolfowitz
    band for an empirical CDF of ``n`` samples."""
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def central_coverage(ranks: IntArray, num_samples: int, levels: FloatArray) -> FloatArray:
    """Empirical coverage of central marginal credible intervals.

    Returns shape ``(len(levels), dim)``.
    """
    u = ranks / num_samples
    lo = (1.0 - levels)[:, None, None] / 2.0
    hi = (1.0 + levels)[:, None, None] / 2.0
    inside = (u[None] >= lo) & (u[None] <= hi)
    return np.asarray(inside.mean(axis=1), dtype=np.float64)


def hpd_levels(log_q_samples: FloatArray, log_q_true: FloatArray) -> FloatArray:
    """Smallest HPD credibility level whose region contains the truth.

    ``alpha_i`` = fraction of posterior draws with higher density than the
    true parameter. For a calibrated posterior ``alpha`` ~ U(0, 1).

    Args:
        log_q_samples: Log densities of the posterior draws, ``(L, n)``.
        log_q_true: Log densities of the true parameters, ``(n,)``.
    """
    return np.asarray(np.mean(log_q_samples > log_q_true[None, :], axis=0), dtype=np.float64)


def hpd_coverage(alpha: FloatArray, levels: FloatArray) -> FloatArray:
    """Expected coverage of HPD regions: fraction of ``alpha < gamma``."""
    return np.asarray((alpha[None, :] < levels[:, None]).mean(axis=1), dtype=np.float64)


def coverage_band(n: int, levels: FloatArray, alpha: float = 0.05) -> tuple[FloatArray, FloatArray]:
    """Pointwise binomial (1 - alpha) band for an empirical coverage at each level."""
    lo = stats.binom.ppf(alpha / 2, n, levels) / n
    hi = stats.binom.ppf(1 - alpha / 2, n, levels) / n
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def recovery_metrics(
    samples: FloatArray, theta: FloatArray, prior_bounds: FloatArray
) -> dict[str, FloatArray]:
    """Parameter-recovery statistics over many (theta*, x) pairs.

    * ``contraction`` = 1 - Var_post / Var_prior (0: nothing learned, 1: exact),
      averaged over observations (Schad, Betancourt & Vasishth 2021);
    * ``z_mean``, ``z_std``: mean and SD of the posterior z-score
      (E_post[theta] - theta*) / SD_post[theta]; ~ (0, 1) if calibrated;
    * ``nrmse``: RMSE of the posterior mean, in units of the prior SD;
    * ``r``: Pearson correlation between posterior mean and truth;
    * ``coverage95``: fraction of central 95 % intervals containing theta*.

    All arrays have shape ``(dim,)``.
    """
    mean = samples.mean(axis=0)
    sd = samples.std(axis=0)
    prior_var = (prior_bounds[:, 1] - prior_bounds[:, 0]) ** 2 / 12.0
    z = (mean - theta) / sd
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    r = np.array([np.corrcoef(mean[:, j], theta[:, j])[0, 1] for j in range(theta.shape[1])])
    return {
        "contraction": np.asarray(1.0 - (sd**2 / prior_var).mean(axis=0)),
        "z_mean": np.asarray(z.mean(axis=0)),
        "z_std": np.asarray(z.std(axis=0)),
        "nrmse": np.asarray(np.sqrt(((mean - theta) ** 2).mean(axis=0) / prior_var)),
        "r": r,
        "coverage95": np.asarray(((theta >= lo) & (theta <= hi)).mean(axis=0)),
    }


def marginal_summary(samples: FloatArray, truth: FloatArray) -> list[dict[str, Any]]:
    """Per-parameter summary of a single posterior (worked recovery).

    ``percentile`` is the posterior CDF at the true value; it lies in
    [2.5, 97.5] iff the truth is inside the central 95 % credible interval.
    """
    q = np.quantile(samples, [0.025, 0.05, 0.5, 0.95, 0.975], axis=0)
    out: list[dict[str, Any]] = []
    for j in range(samples.shape[1]):
        pct = 100.0 * float(np.mean(samples[:, j] < truth[j]))
        out.append(
            {
                "truth": float(truth[j]),
                "mean": float(samples[:, j].mean()),
                "sd": float(samples[:, j].std()),
                "median": float(q[2, j]),
                "ci95": [float(q[0, j]), float(q[4, j])],
                "ci90": [float(q[1, j]), float(q[3, j])],
                "percentile": pct,
                "inside_ci95": bool(q[0, j] <= truth[j] <= q[4, j]),
            }
        )
    return out


def predictive_check(x_obs: FloatArray, x_pred: FloatArray) -> list[dict[str, float]]:
    """Posterior (or prior) predictive check, one entry per summary statistic.

    ``quantile`` is the mid-p predictive CDF at the observed value (ties
    counted half, which matters for the integer spike count); ``p_value`` is
    the two-sided predictive p-value ``2 min(q, 1 - q)``; ``z`` is the
    observed value standardized by the predictive mean and SD.
    """
    out: list[dict[str, float]] = []
    for j in range(x_obs.shape[0]):
        col = x_pred[:, j]
        q = float(np.mean(col < x_obs[j]) + 0.5 * np.mean(col == x_obs[j]))
        sd = float(col.std())
        out.append(
            {
                "observed": float(x_obs[j]),
                "pred_mean": float(col.mean()),
                "pred_sd": sd,
                "quantile": q,
                "p_value": float(2.0 * min(q, 1.0 - q)),
                "z": float((x_obs[j] - col.mean()) / sd) if sd > 0 else 0.0,
            }
        )
    return out
