"""Amortized neural posterior estimation (NPE) with the ``sbi`` toolkit.

A conditional normalizing flow q_phi(theta | x) is trained once on pairs
(theta_i, x_i) drawn from the joint p(theta) p(x | theta) by maximizing
sum_i log q_phi(theta_i | x_i). Its optimum is the true posterior p(theta | x)
for *every* x (Papamakarios & Murray 2016; Greenberg et al. 2019), so after
training, inference on a new observation is a single forward pass -- no
further simulations and no MCMC.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sbi.inference import NPE, DirectPosterior
from sbi.neural_nets import posterior_nn
from torch import nn

from neuro_sbi.config import TrainingConfig
from neuro_sbi.features import FEATURE_NAMES
from neuro_sbi.params import ParameterSpace
from neuro_sbi.simulator import SimulationSettings

FloatArray = NDArray[np.float64]

# Heavy-tailed features that are log10-transformed before standardization
# (resting SD spans ~0.05 mV in quiescent cells to >10 mV in spontaneously
# firing ones).
LOG_FEATURES: tuple[str, ...] = ("resting_std",)
LOG_FLOOR = 1e-3


class FeatureStandardizer(nn.Module):
    """Fixed preprocessing of the summary statistics, stored inside the network.

    ``x -> (T(x) - mean) / std`` with ``T = log10`` on :data:`LOG_FEATURES`.
    Because it is part of the estimator (as its embedding net), the saved
    posterior accepts raw feature vectors.
    """

    mean: torch.Tensor
    std: torch.Tensor
    log_mask: torch.Tensor

    def __init__(self, names: Sequence[str] = FEATURE_NAMES) -> None:
        super().__init__()
        n = len(names)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))
        self.register_buffer("log_mask", torch.tensor([nm in LOG_FEATURES for nm in names]))
        # sbi's device check inspects the embedding net's first parameter, so the
        # (otherwise parameter-free) module carries one frozen scalar.
        self.device_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.log_mask, torch.log10(x.clamp_min(LOG_FLOOR)), x)

    def fit(self, x: torch.Tensor) -> FeatureStandardizer:
        t = self.transform(x)
        self.mean.copy_(t.mean(dim=0))
        self.std.copy_(t.std(dim=0).clamp_min(1e-6))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.transform(x) - self.mean) / self.std


class NullTracker:
    """``sbi`` experiment tracker that records nothing.

    The default tracker writes TensorBoard event files to ``./sbi-logs``; the
    loss curves are already returned by :func:`train_npe`.
    """

    @property
    def log_dir(self) -> str | None:
        return None

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def add_figure(self, name: str, figure: Any, step: int | None = None) -> None:
        pass

    def flush(self) -> None:
        pass


def _as_tensor(a: FloatArray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def _builder(cfg: TrainingConfig, embedding: FeatureStandardizer) -> Any:
    kwargs: dict[str, Any] = {
        "model": cfg.density_estimator,
        "hidden_features": cfg.hidden_features,
        "num_transforms": cfg.num_transforms,
        "z_score_theta": "independent",
        "z_score_x": "none",  # handled by FeatureStandardizer
        "embedding_net": embedding,
    }
    if cfg.density_estimator == "nsf":
        kwargs["num_bins"] = cfg.num_bins
    return posterior_nn(**kwargs)


def train_npe(
    theta: FloatArray,
    x: FloatArray,
    space: ParameterSpace,
    cfg: TrainingConfig,
    seed: int,
) -> tuple[DirectPosterior, dict[str, Any]]:
    """Train a neural posterior estimator.

    Args:
        theta: Parameters in inference space, shape ``(n, dim)``.
        x: Summary statistics, shape ``(n, n_features)``.
        space: Parameter space (defines the prior).
        cfg: Network and optimizer settings.
        seed: Seed for network initialization, data split and mini-batching.

    Returns:
        The amortized posterior and a training summary (loss curves, epochs).
    """
    if not np.all(np.isfinite(x)):
        raise ValueError("Summary statistics contain non-finite values")
    torch.manual_seed(seed)
    x_t = _as_tensor(x)
    inference = NPE(
        prior=space.prior(),
        density_estimator=_builder(cfg, FeatureStandardizer().fit(x_t)),
        device="cpu",
        tracker=NullTracker(),
        show_progress_bars=False,
    )
    with warnings.catch_warnings():
        # sbi warns about outliers for raw-x z-scoring, which we do not use.
        warnings.filterwarnings("ignore", message=".*extreme outliers.*")
        inference.append_simulations(_as_tensor(theta), x_t)
    # Learning-rate annealing on plateau: train to early-stopping convergence,
    # then resume with each smaller rate in ``cfg.lr_decay``. Every phase keeps
    # the same train/validation split and restarts its patience counter;
    # sbi restores the best-validation weights at the end of each phase.
    phase_boundaries: list[int] = []
    for phase, lr in enumerate((cfg.learning_rate, *cfg.lr_decay)):
        if phase > 0:
            if not hasattr(inference, "_epochs_since_last_improvement"):
                raise RuntimeError("Unsupported sbi version: cannot resume training")
            for group in inference.optimizer.param_groups:
                group["lr"] = lr
            inference._epochs_since_last_improvement = 0
        estimator = inference.train(
            training_batch_size=cfg.training_batch_size,
            learning_rate=lr,
            validation_fraction=cfg.validation_fraction,
            stop_after_epochs=cfg.stop_after_epochs,
            max_num_epochs=(inference.epoch if phase > 0 else 0) + cfg.max_num_epochs,
            resume_training=phase > 0,
            show_train_summary=False,
        )
        phase_boundaries.append(len(inference.summary["training_loss"]))
    posterior = inference.build_posterior(estimator)
    summary = inference.summary
    info: dict[str, Any] = {
        "epochs_trained": len(summary["training_loss"]),
        "phase_end_epochs": phase_boundaries,
        "learning_rates": [cfg.learning_rate, *cfg.lr_decay],
        "best_validation_loss": float(summary["best_validation_loss"][-1]),
        "training_loss": [float(v) for v in summary["training_loss"]],
        "validation_loss": [float(v) for v in summary["validation_loss"]],
        "num_parameters": int(sum(p.numel() for p in estimator.parameters())),
    }
    return posterior, info


@dataclass(frozen=True)
class LoadedPosterior:
    """A trained posterior together with everything needed to use it."""

    posterior: DirectPosterior
    space: ParameterSpace
    simulation: SimulationSettings
    training: TrainingConfig


def save_posterior(
    posterior: DirectPosterior,
    path: str | Path,
    space: ParameterSpace,
    training: TrainingConfig,
    simulation: SimulationSettings,
) -> None:
    """Save the trained estimator as a plain ``state_dict`` plus metadata.

    A state dict (rather than a pickled object) is loaded with
    ``torch.load(weights_only=True)``, which cannot execute arbitrary code.
    The simulation protocol is stored too: the posterior is only valid for
    observations recorded with the same protocol.
    """
    torch.save(
        {
            "state_dict": posterior.posterior_estimator.state_dict(),
            "inferred": list(space.names),
            "training": asdict(training),
            "simulation": asdict(simulation),
            "features": list(FEATURE_NAMES),
        },
        Path(path),
    )


def load_posterior(path: str | Path) -> LoadedPosterior:
    """Rebuild a posterior saved with :func:`save_posterior`."""
    blob = torch.load(Path(path), weights_only=True)
    if list(blob["features"]) != list(FEATURE_NAMES):
        raise ValueError("Posterior was trained on a different feature set")
    space = ParameterSpace(tuple(blob["inferred"]))
    training = TrainingConfig(**{**blob["training"], "lr_decay": tuple(blob["training"].get("lr_decay", ()))})
    prior = space.prior()
    # The builder needs example batches to infer shapes; all standardization
    # statistics are buffers and are overwritten by the saved state dict.
    dummy_theta = prior.sample((32,))
    dummy_x = torch.rand(32, len(FEATURE_NAMES))
    estimator = _builder(training, FeatureStandardizer())(dummy_theta, dummy_x)
    estimator.load_state_dict(blob["state_dict"])
    estimator.eval()
    return LoadedPosterior(
        posterior=DirectPosterior(posterior_estimator=estimator, prior=prior),
        space=space,
        simulation=SimulationSettings(**blob["simulation"]),
        training=training,
    )


def sample(posterior: DirectPosterior, x: FloatArray, n: int) -> FloatArray:
    """Draw ``n`` posterior samples for a single observation ``x``."""
    s = posterior.sample((n,), x=_as_tensor(x).reshape(1, -1), show_progress_bars=False)
    return np.asarray(s.detach().cpu().numpy(), dtype=np.float64)


def sample_batched(posterior: DirectPosterior, xs: FloatArray, n: int) -> FloatArray:
    """Draw ``n`` posterior samples for each of many observations.

    Returns an array of shape ``(n, num_observations, dim)``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_sampling_batch_size.*")
        s = posterior.sample_batched((n,), x=_as_tensor(xs), show_progress_bars=False)
    return np.asarray(s.detach().cpu().numpy(), dtype=np.float64)


@torch.no_grad()
def log_prob_unnormalized(posterior: DirectPosterior, theta: FloatArray, xs: FloatArray) -> FloatArray:
    """Flow log-density log q(theta | x), *without* the prior-truncation
    normalizer (which is constant in theta and cancels in density ranks).

    Args:
        theta: Shape ``(n_samples, n_obs, dim)``.
        xs: Shape ``(n_obs, n_features)``.

    Returns:
        Shape ``(n_samples, n_obs)``.
    """
    est = posterior.posterior_estimator
    lp = est.log_prob(_as_tensor(theta), condition=_as_tensor(xs))
    return np.asarray(lp.cpu().numpy(), dtype=np.float64)
