"""Amortized neural posterior estimation (NPE) with the ``sbi`` toolkit.

A conditional normalizing flow q_phi(theta | x) is trained once on pairs
(theta_i, x_i) drawn from the joint p(theta) p(x | theta) by maximizing
sum_i log q_phi(theta_i | x_i). Its optimum is the true posterior p(theta | x)
for *every* x (Papamakarios & Murray 2016; Greenberg et al. 2019), so after
training, inference on a new observation is a single forward pass -- no
further simulations and no MCMC.

A single flow trained on a finite budget tends to be over-confident
(Hermans et al. 2022). Several flows trained from different random
initializations are therefore combined into an equal-weight mixture, a *deep
ensemble* (Lakshminarayanan et al. 2017): where the members disagree, the
mixture is wider than any of them.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sbi.inference import NPE, DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.utils import within_support
from torch import Tensor, nn

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

# Rejection sampling to the prior support (see :func:`sample_batched`).
MAX_PROPOSALS_PER_ROUND = 200_000
MAX_PROPOSALS_PER_OBSERVATION = 20_000_000


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


class FlowEnsemble(ConditionalDensityEstimator):  # type: ignore[misc]  # sbi is untyped
    """Equal-weight mixture of conditional density estimators.

    ``q(theta | x) = (1/K) sum_k q_k(theta | x)``. Each draw picks a member
    uniformly at random and samples from it, so samples are exact i.i.d. draws
    from the mixture; ``log_prob`` is the log of the mixture density. Wrapped
    in an ``sbi`` ``DirectPosterior``, the mixture is truncated to the prior
    support as a whole by rejection sampling, exactly as a single flow is.
    """

    def __init__(self, members: Sequence[ConditionalDensityEstimator]) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        first = members[0]
        if any(m.input_shape != first.input_shape for m in members) or any(
            m.condition_shape != first.condition_shape for m in members
        ):
            raise ValueError("Ensemble members must share input and condition shapes")
        super().__init__(nn.ModuleList(members), first.input_shape, first.condition_shape)

    @property
    def members(self) -> list[ConditionalDensityEstimator]:
        return list(self.net)

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs: Any) -> Tensor:
        lps = torch.stack([m.log_prob(input, condition, **kwargs) for m in self.net])
        return torch.logsumexp(lps, dim=0) - math.log(len(self.net))

    def loss(self, input: Tensor, condition: Tensor, **kwargs: Any) -> Tensor:
        return -self.log_prob(input, condition, **kwargs)

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs: Any) -> Tensor:
        n = torch.Size(sample_shape).numel()
        batch = condition.shape[0]
        labels = torch.randint(len(self.net), (n, batch), device=condition.device)
        out = torch.empty(n, batch, *self.input_shape, device=condition.device)
        for k, member in enumerate(self.net):
            mask = labels == k
            need = int(mask.sum(dim=0).max())
            if need == 0:
                continue
            # Draw as many samples per observation as the most-requested
            # observation needs; observation b uses the first count_b of them.
            draws = member.sample(torch.Size([need]), condition, **kwargs)
            rank = torch.cumsum(mask, dim=0) - 1
            i_idx, b_idx = mask.nonzero(as_tuple=True)
            out[i_idx, b_idx] = draws[rank[i_idx, b_idx], b_idx]
        return out.reshape(*sample_shape, batch, *self.input_shape)


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


def _train_member(
    job: tuple[FloatArray, FloatArray, tuple[str, ...], TrainingConfig, int],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Train one ensemble member on a single CPU thread.

    One thread is both the fastest setting for these small networks and what
    makes a member's weights independent of how many members train in
    parallel.
    """
    theta, x, inferred, cfg, seed = job
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        posterior, info = train_npe(theta, x, ParameterSpace(inferred), cfg, seed)
    finally:
        torch.set_num_threads(threads)
    return posterior.posterior_estimator.state_dict(), info | {"seed": seed}


def train_ensemble(
    theta: FloatArray,
    x: FloatArray,
    space: ParameterSpace,
    cfg: TrainingConfig,
    seeds: Sequence[int],
    workers: int = 1,
) -> tuple[DirectPosterior, list[dict[str, Any]]]:
    """Train ``len(seeds)`` flows that differ only in their random seed.

    The members share the training set; the seed sets the initial weights,
    the train/validation split and the mini-batch order. Members train in up
    to ``workers`` processes and the result does not depend on ``workers``.

    Returns:
        The posterior (a single flow if there is one seed, otherwise their
        :class:`FlowEnsemble`) and each member's training summary.
    """
    jobs = [(theta, x, space.names, cfg, int(seed)) for seed in seeds]
    if not jobs:
        raise ValueError("At least one seed is required")
    if workers <= 1 or len(jobs) == 1:
        results = [_train_member(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")
        ) as pool:
            results = list(pool.map(_train_member, jobs))
    posterior = build_posterior(space, cfg, [state for state, _ in results])
    return posterior, [info for _, info in results]


def build_posterior(
    space: ParameterSpace, training: TrainingConfig, state_dicts: Sequence[dict[str, Tensor]]
) -> DirectPosterior:
    """Rebuild a (single-flow or ensemble) posterior from member state dicts."""
    prior = space.prior()
    estimators = []
    # Building a network draws initial weights from the global torch RNG;
    # fork it so that rebuilding a posterior has no side effects.
    with torch.random.fork_rng():
        for state in state_dicts:
            # The builder needs example batches to infer shapes; all
            # standardization statistics are buffers, overwritten by the state.
            dummy_theta = prior.sample((32,))
            dummy_x = torch.rand(32, len(FEATURE_NAMES))
            est = _builder(training, FeatureStandardizer())(dummy_theta, dummy_x)
            est.load_state_dict(state)
            est.eval()
            estimators.append(est)
    estimator = estimators[0] if len(estimators) == 1 else FlowEnsemble(estimators)
    return DirectPosterior(posterior_estimator=estimator, prior=prior)


def ensemble_members(posterior: DirectPosterior) -> list[DirectPosterior]:
    """The individual flows of an ensemble posterior, each as a posterior
    with the same prior (a single-flow posterior is its own only member)."""
    est = posterior.posterior_estimator
    if isinstance(est, FlowEnsemble):
        return [DirectPosterior(posterior_estimator=m, prior=posterior.prior) for m in est.members]
    return [posterior]


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
    """Save the trained flow(s) as plain state dicts plus metadata.

    A state dict (rather than a pickled object) is loaded with
    ``torch.load(weights_only=True)``, which cannot execute arbitrary code.
    The simulation protocol is stored too: the posterior is only valid for
    observations recorded with the same protocol.
    """
    torch.save(
        {
            "state_dicts": [m.posterior_estimator.state_dict() for m in ensemble_members(posterior)],
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
    # Files written before ensembles were supported hold a single "state_dict".
    states = blob["state_dicts"] if "state_dicts" in blob else [blob["state_dict"]]
    if len(states) != training.ensemble_size:
        raise ValueError(f"{path}: {len(states)} flows stored, ensemble_size is {training.ensemble_size}")
    return LoadedPosterior(
        posterior=build_posterior(space, training, states),
        space=space,
        simulation=SimulationSettings(**blob["simulation"]),
        training=training,
    )


def sample(posterior: DirectPosterior, x: FloatArray, n: int) -> FloatArray:
    """Draw ``n`` posterior samples for a single observation ``x``."""
    return sample_batched(posterior, np.asarray(x).reshape(1, -1), n)[:, 0]


@torch.no_grad()
def sample_batched(posterior: DirectPosterior, xs: FloatArray, n: int) -> FloatArray:
    """Draw ``n`` posterior samples for each of many observations.

    Samples are drawn from the flow (or ensemble) and rejected outside the
    prior support, so they follow the flow truncated to the prior, as in
    ``sbi``. Unlike ``sbi``'s batched sampler, each round proposes only for
    the observations that still lack samples, with a batch sized by their
    acceptance rate so far. One observation whose posterior lies mostly
    outside the prior therefore does not hold up all the others.

    Returns an array of shape ``(n, num_observations, dim)``.
    """
    est = posterior.posterior_estimator
    x = _as_tensor(xs).reshape(np.shape(xs)[0], -1)
    num_obs = x.shape[0]
    out = torch.empty(n, num_obs, *est.input_shape)
    filled = torch.zeros(num_obs, dtype=torch.long)
    proposed = torch.zeros(num_obs, dtype=torch.long)
    accepted = torch.zeros(num_obs, dtype=torch.long)
    todo = torch.arange(num_obs)
    while todo.numel() > 0:
        rate = (accepted[todo] + 1) / (proposed[todo] + 1)
        wanted = float(torch.max(1.1 * (n - filled[todo]) / rate))
        per_obs = int(min(max(wanted, 1.0), max(1, MAX_PROPOSALS_PER_ROUND // todo.numel())))
        cand = est.sample(torch.Size([per_obs]), x[todo])  # (per_obs, len(todo), dim)
        ok = within_support(posterior.prior, cand.reshape(-1, cand.shape[-1])).reshape(per_obs, -1)
        # Accepted candidates fill the next free slots, in proposal order.
        slot = torch.cumsum(ok, dim=0) - 1 + filled[todo]
        i_idx, j_idx = (ok & (slot < n)).nonzero(as_tuple=True)
        out[slot[i_idx, j_idx], todo[j_idx]] = cand[i_idx, j_idx]
        proposed[todo] += per_obs
        accepted[todo] += ok.sum(dim=0)
        filled[todo] = torch.clamp(filled[todo] + ok.sum(dim=0), max=n)
        todo = todo[filled[todo] < n]
        if todo.numel() and int(proposed[todo].max()) > MAX_PROPOSALS_PER_OBSERVATION:
            worst = int(todo[torch.argmax(proposed[todo])])
            raise RuntimeError(
                f"Observation {worst}: only {int(accepted[worst])} of {int(proposed[worst])} posterior "
                "proposals fall inside the prior support"
            )
    return np.asarray(out.cpu().numpy(), dtype=np.float64)


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
