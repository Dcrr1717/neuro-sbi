"""Figures for the SBI pipeline (matplotlib, non-interactive backend)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from neuro_sbi.diagnostics import (
    coverage_band,
    dkw_epsilon,
    histogram_band,
    rank_bin_probabilities,
)

FloatArray = NDArray[np.float64]

# Palette (validated categorical order; see README "Figures").
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BAND = "#e8e7e1"
POSTERIOR = "#2a78d6"  # slot 1 (blue)
TRUTH = "#eb6834"  # slot 2 (orange)
PRIOR = "#a9a8a2"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.size": 9,
            "lines.linewidth": 1.5,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def pairplot(
    samples: FloatArray,
    truth: FloatArray,
    bounds: FloatArray,
    labels: Sequence[str],
    path: str | Path,
    title: str = "",
    bins: int = 50,
) -> None:
    """Posterior pairplot. Axis limits are the prior bounds, so the visible
    shrinkage of each marginal is the information gained from the data."""
    set_style()
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(1.9 * d, 1.9 * d), squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                edges = np.linspace(bounds[i, 0], bounds[i, 1], bins + 1)
                ax.hist(samples[:, i], bins=edges, density=True, color=POSTERIOR, alpha=0.35)
                ax.hist(
                    samples[:, i],
                    bins=edges,
                    density=True,
                    histtype="step",
                    color=POSTERIOR,
                    linewidth=1.2,
                )
                lo, hi = np.quantile(samples[:, i], [0.025, 0.975])
                for q in (lo, hi):
                    ax.axvline(q, color=POSTERIOR, linewidth=0.8, linestyle=":")
                ax.axvline(truth[i], color=TRUTH, linewidth=2.0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.set_xlim(bounds[i])
            else:
                ax.hist2d(
                    samples[:, j],
                    samples[:, i],
                    bins=bins,
                    range=[list(bounds[j]), list(bounds[i])],
                    cmap=SEQUENTIAL,
                )
                ax.plot(
                    truth[j],
                    truth[i],
                    marker="o",
                    markersize=7,
                    color=TRUTH,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                )
                ax.set_xlim(bounds[j])
                ax.set_ylim(bounds[i])
            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0 or i == 0:
                if i != j:
                    ax.set_yticklabels([])
    handles = [
        Line2D([], [], color=POSTERIOR, linewidth=6, alpha=0.5, label="posterior"),
        Line2D([], [], color=POSTERIOR, linestyle=":", label="95 % credible interval"),
        Line2D([], [], color=TRUTH, linewidth=2, label="ground truth"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, path)


def ppc_traces(
    t: FloatArray,
    v_obs: FloatArray,
    v_post: FloatArray,
    v_prior: FloatArray,
    stimulus: FloatArray,
    t_on: float,
    t_off: float,
    path: str | Path,
) -> None:
    """Observed voltage trace against posterior- and prior-predictive traces."""
    set_style()
    k = v_post.shape[0]
    fig = plt.figure(figsize=(10, 1.75 * (k + 1) + 0.6))
    gs = fig.add_gridspec(k + 1, 2, hspace=0.55, wspace=0.08, top=0.93)
    ylim = (-95.0, 60.0)

    def _panel(ax: Axes, v: FloatArray, color: str, label: str, last: bool) -> None:
        ax.axvspan(t_on, t_off, color=BAND, zorder=0, linewidth=0)
        ax.plot(t, v, color=color, linewidth=0.9)
        ax.set_ylim(ylim)
        ax.set_xlim(t[0], t[-1])
        ax.set_title(label, loc="left", fontsize=8.5, color=INK_2, pad=3)
        if not last:
            ax.set_xticklabels([])

    ax_obs = fig.add_subplot(gs[0, 0])
    _panel(ax_obs, v_obs, INK, "observed: ground-truth parameters, held-out noise seed", False)
    ax_obs.set_ylabel("V [mV]")
    ax_stim = fig.add_subplot(gs[0, 1])
    ax_stim.axvspan(t_on, t_off, color=BAND, zorder=0, linewidth=0)
    ax_stim.plot(t, stimulus, color=INK_2, linewidth=1.2)
    ax_stim.set_xlim(t[0], t[-1])
    i_max = float(np.max(stimulus))
    ax_stim.set_ylim(-0.1 * i_max - 0.1, 1.3 * i_max + 0.1)
    ax_stim.set_title("injected current [µA/cm²]", loc="left", fontsize=8.5, color=INK_2, pad=3)
    ax_stim.set_xticklabels([])
    for r in range(k):
        last = r == k - 1
        axl = fig.add_subplot(gs[r + 1, 0])
        _panel(axl, v_post[r], POSTERIOR, f"posterior predictive #{r + 1}", last)
        axl.set_ylabel("V [mV]")
        axr = fig.add_subplot(gs[r + 1, 1])
        _panel(axr, v_prior[r], PRIOR, f"prior predictive #{r + 1}", last)
        axr.set_yticklabels([])
        if last:
            axl.set_xlabel("time [ms]  (shaded: current step)")
            axr.set_xlabel("time [ms]  (shaded: current step)")
    fig.suptitle(
        "Posterior predictive check: simulations from posterior (left) vs prior (right) draws",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=11,
    )
    _save(fig, path)


def ppc_features(
    x_obs: FloatArray,
    x_post: FloatArray,
    x_prior: FloatArray,
    names: Sequence[str],
    units: Sequence[str],
    quantiles: Sequence[float],
    path: str | Path,
) -> None:
    """Predictive distributions of every summary statistic."""
    set_style()
    nf = len(names)
    ncol = 4
    nrow = int(np.ceil(nf / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes.flat[j]
        if j >= nf:
            ax.axis("off")
            continue
        # Zoom on the posterior predictive (plus the observation); the prior
        # predictive is drawn with the same density normalization over its full
        # sample, so only its in-window part is visible.
        lo, hi = np.quantile(x_post[:, j], [0.005, 0.995])
        lo, hi = min(lo, x_obs[j]), max(hi, x_obs[j])
        full = np.ptp(x_prior[:, j]) or 1.0
        width = max(hi - lo, 0.02 * full)
        lo, hi = lo - 0.5 * width, hi + 0.5 * width
        if names[j] == "spike_count":
            edges = np.arange(np.floor(lo), np.ceil(hi) + 2) - 0.5
        else:
            edges = np.linspace(lo, hi, 41)
        bw = float(edges[1] - edges[0])
        for data, kw in (
            (x_prior[:, j], {"histtype": "step", "color": PRIOR, "linewidth": 1.2}),
            (x_post[:, j], {"color": POSTERIOR, "alpha": 0.55}),
        ):
            ax.hist(data, bins=edges, weights=np.full(data.size, 1.0 / (data.size * bw)), **kw)
        ax.axvline(x_obs[j], color=INK, linewidth=1.8)
        ax.set_xlim(edges[0], edges[-1])
        unit = f" [{units[j]}]" if units[j] else ""
        ax.set_title(f"{names[j]}{unit}", loc="left", fontsize=9)
        ax.set_title(f"q = {quantiles[j]:.2f}", loc="right", fontsize=8, color=INK_2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    handles = [
        Line2D([], [], color=PRIOR, linewidth=1.2, label="prior predictive"),
        Line2D([], [], color=POSTERIOR, linewidth=6, alpha=0.55, label="posterior predictive"),
        Line2D([], [], color=INK, linewidth=1.8, label="observed"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.text(
        0.5,
        -0.01,
        "x-axes zoomed on the posterior predictive; q = predictive quantile of the observed value",
        ha="center",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, path)


def sbc_rank_histograms(
    ranks: NDArray[np.int64],
    num_samples: int,
    n_bins: int,
    labels: Sequence[str],
    pvalues: Sequence[float],
    path: str | Path,
) -> None:
    """SBC rank histograms with the 99 % pointwise binomial band (Talts et al. 2018)."""
    set_style()
    n, d = ranks.shape
    ncol = min(3, d)
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow), squeeze=False)
    probs = rank_bin_probabilities(num_samples, n_bins)
    lo, hi = histogram_band(n, n_bins, alpha=0.01)
    for j in range(nrow * ncol):
        ax = axes.flat[j]
        if j >= d:
            ax.axis("off")
            continue
        counts = np.bincount((ranks[:, j] * n_bins) // (num_samples + 1), minlength=n_bins)
        ax.axhspan(lo, hi, color=BAND, linewidth=0, zorder=0)
        ax.axhline(n * probs.mean(), color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
        ax.bar(
            np.arange(n_bins),
            counts,
            width=0.85,
            color=POSTERIOR,
            zorder=2,
        )
        ax.set_title(f"{labels[j]}\nKS p = {pvalues[j]:.2f}", loc="left", fontsize=9)
        ax.set_xticks([0, n_bins - 1])
        ax.set_xticklabels(["0", f"L={num_samples}"])
        ax.set_ylim(0, max(hi, float(np.max(counts))) * 1.15)
        if j % ncol == 0:
            ax.set_ylabel("count")
    fig.suptitle(
        f"SBC rank histograms ({n} simulations; band: 99 % binomial interval under uniformity)",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, path)


def sbc_ecdf(ranks: NDArray[np.int64], num_samples: int, labels: Sequence[str], path: str | Path) -> None:
    """ECDF of normalized ranks minus the uniform CDF, with a simultaneous
    95 % Dvoretzky-Kiefer-Wolfowitz band."""
    set_style()
    n, d = ranks.shape
    eps = dkw_epsilon(n, alpha=0.05)
    grid = np.linspace(0, 1, 201)
    ncol = min(3, d)
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes.flat[j]
        if j >= d:
            ax.axis("off")
            continue
        u = np.sort((ranks[:, j] + 0.5) / (num_samples + 1))
        ecdf = np.searchsorted(u, grid, side="right") / n
        ax.fill_between(grid, -eps, eps, color=BAND, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.plot(grid, ecdf - grid, color=POSTERIOR)
        ax.set_ylim(-max(2.2 * eps, 0.05), max(2.2 * eps, 0.05))
        ax.set_title(labels[j], loc="left", fontsize=9)
        if j % ncol == 0:
            ax.set_ylabel("ECDF − uniform")
        if j >= d - ncol:
            ax.set_xlabel("normalized rank")
    fig.suptitle("SBC: ECDF difference (band: simultaneous 95 % DKW band)", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def coverage_plot(
    levels: FloatArray,
    marginal: FloatArray,
    joint: FloatArray,
    n: int,
    labels: Sequence[str],
    path: str | Path,
) -> None:
    """Expected coverage of marginal central intervals and joint HPD regions."""
    set_style()
    lo, hi = coverage_band(n, levels, alpha=0.05)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax in axes:
        ax.fill_between(levels, lo, hi, color=BAND, linewidth=0)
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("credibility level")
        ax.grid(True)
    for j, lab in enumerate(labels):
        axes[0].plot(
            levels, marginal[:, j], color=CATEGORICAL[j % len(CATEGORICAL)], linewidth=1.3, label=lab
        )
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_title("Marginal central credible intervals", loc="left")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[1].plot(levels, joint, color=POSTERIOR, linewidth=2.0)
    axes[1].set_title("Joint highest-posterior-density regions", loc="left")
    axes[1].text(
        0.98,
        0.04,
        "above diagonal: conservative\nbelow diagonal: over-confident",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_2,
    )
    fig.suptitle(
        f"Expected coverage on {n} held-out simulations (band: 95 % binomial interval)",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, path)


def recovery_plot(
    theta: FloatArray,
    samples: FloatArray,
    bounds: FloatArray,
    labels: Sequence[str],
    contraction: FloatArray,
    r: FloatArray,
    path: str | Path,
    spiking: NDArray[np.bool_] | None = None,
    max_points: int = 300,
) -> None:
    """Posterior median and 90 % credible interval vs ground truth.

    If ``spiking`` is given, cells that fired no AP during the stimulus are
    drawn in grey: for them the spike-generating parameters are not
    identifiable and their posteriors should revert towards the prior.
    """
    set_style()
    d = theta.shape[1]
    m = min(max_points, theta.shape[0])
    med = np.median(samples[:, :m], axis=0)
    lo, hi = np.quantile(samples[:, :m], [0.05, 0.95], axis=0)
    fired = np.ones(m, dtype=bool) if spiking is None else np.asarray(spiking[:m], dtype=bool)
    groups = ((fired, POSTERIOR, "fired ≥ 1 AP"), (~fired, PRIOR, "silent (no AP)"))
    ncol = min(3, d)
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes.flat[j]
        if j >= d:
            ax.axis("off")
            continue
        ax.plot(bounds[j], bounds[j], color=MUTED, linestyle="--", linewidth=0.8)
        for sel, color, _ in groups:
            ax.vlines(theta[:m, j][sel], lo[sel, j], hi[sel, j], color=color, alpha=0.25, linewidth=0.8)
            ax.plot(theta[:m, j][sel], med[sel, j], "o", markersize=3, color=color)
        ax.set_xlim(bounds[j])
        ax.set_ylim(bounds[j])
        ax.set_aspect("equal")
        ax.set_title(f"{labels[j]}\ncontraction {contraction[j]:.2f}, r = {r[j]:.2f}", loc="left", fontsize=9)
        ax.set_xlabel("true")
        if j % ncol == 0:
            ax.set_ylabel("posterior median (90 % CI)")
    fig.suptitle(f"Amortized parameter recovery on {m} held-out simulations", x=0.02, ha="left", fontsize=10)
    if spiking is not None:
        handles = [
            Line2D([], [], color=c, marker="o", linestyle="", markersize=5, label=lab) for _, c, lab in groups
        ]
        fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout()
    _save(fig, path)


def training_curve(
    train: Sequence[float],
    val: Sequence[float],
    path: str | Path,
    phase_ends: Sequence[int] = (),
    learning_rates: Sequence[float] = (),
) -> None:
    """Mean log-density per epoch; dotted lines mark learning-rate decays."""
    set_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ep = np.arange(1, len(train) + 1)
    ax.plot(ep, -np.asarray(train), color=CATEGORICAL[0], label="training")
    ax.plot(np.arange(1, len(val) + 1), -np.asarray(val), color=CATEGORICAL[1], label="validation")
    lo = float(np.quantile(-np.asarray(val), 0.05))
    hi = float(np.max(-np.asarray(val)))
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
    for end, lr in zip(phase_ends[:-1], learning_rates[1:], strict=False):
        ax.axvline(end + 0.5, color=MUTED, linestyle=":", linewidth=1.0)
        ax.text(end + 1.5, lo - 0.1 * (hi - lo), f"lr → {lr:g}", fontsize=7.5, color=INK_2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean log q(θ | x)")
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")
    ax.set_title("NPE training curve", loc="left")
    fig.tight_layout()
    _save(fig, path)
