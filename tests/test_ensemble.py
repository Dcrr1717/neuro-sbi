"""Tests of the flow ensemble (mixture density, sampling, training) and of
rejection sampling to the prior support."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sbi.inference import DirectPosterior
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.utils import BoxUniform
from scipy import stats
from torch import Tensor, nn

from neuro_sbi import inference
from neuro_sbi.config import TINY, TrainingConfig
from neuro_sbi.features import FEATURE_NAMES, N_FEATURES
from neuro_sbi.inference import (
    FlowEnsemble,
    ensemble_members,
    load_posterior,
    log_prob_unnormalized,
    sample,
    sample_batched,
    train_ensemble,
    train_npe,
)
from neuro_sbi.params import ParameterSpace


class ShiftedGaussian(ConditionalDensityEstimator):  # type: ignore[misc]
    """Test double with a known density: theta | x ~ N(x + shift, I)."""

    def __init__(self, shift: float, dim: int = 2) -> None:
        super().__init__(nn.Identity(), torch.Size([dim]), torch.Size([dim]))
        self.shift = shift

    def _dist(self, condition: Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Independent(torch.distributions.Normal(condition + self.shift, 1.0), 1)

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs: Any) -> Tensor:
        return self._dist(condition).log_prob(input)

    def loss(self, input: Tensor, condition: Tensor, **kwargs: Any) -> Tensor:
        return -self.log_prob(input, condition)

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs: Any) -> Tensor:
        return self._dist(condition).sample(sample_shape)


def _ensemble(*shifts: float) -> FlowEnsemble:
    return FlowEnsemble([ShiftedGaussian(s) for s in shifts])


def test_mixture_log_prob_is_the_equal_weight_mixture_density() -> None:
    ens = _ensemble(-3.0, 3.0)
    x = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    theta = torch.randn(5, 2, 2) * 4

    def comp(shift: float) -> Tensor:  # density of N(x + shift, I) in two dimensions
        return torch.exp(-0.5 * ((theta - x - shift) ** 2).sum(-1)) / (2 * math.pi)

    expected = torch.log(0.5 * comp(-3.0) + 0.5 * comp(3.0))
    torch.testing.assert_close(ens.log_prob(theta, x), expected)


@pytest.mark.parametrize("shifts", [(-8.0, 8.0), (-8.0, 0.0, 8.0)])
def test_mixture_samples_each_member_equally_often(shifts: tuple[float, ...]) -> None:
    torch.manual_seed(0)
    ens = _ensemble(*shifts)
    x = torch.tensor([[0.0, 0.0], [5.0, -5.0], [1.0, 2.0]])
    s = ens.sample(torch.Size([20_000]), x)
    assert s.shape == (20_000, 3, 2)
    # Components are 8 SDs apart, so each draw is attributable to one member.
    member = torch.argmin(torch.stack([(s - x - sh).abs().sum(-1) for sh in shifts]), dim=0)
    for k, sh in enumerate(shifts):
        frac = (member == k).float().mean(dim=0)
        assert torch.all((frac - 1 / len(shifts)).abs() < 0.015), (sh, frac)
        # Draws assigned to a member follow that member's distribution.
        within = (s - x - sh)[member == k]
        assert within.mean().abs() < 0.03 and (within.std() - 1).abs() < 0.03


def test_mixture_sample_shape_and_prior_truncation() -> None:
    torch.manual_seed(1)
    ens = _ensemble(-1.0, 1.0)
    x = torch.zeros(4, 2)
    assert ens.sample(torch.Size([3, 5]), x).shape == (3, 5, 4, 2)
    post = DirectPosterior(posterior_estimator=ens, prior=BoxUniform(-torch.ones(2), torch.ones(2)))
    s = post.sample_batched((500,), x=x, show_progress_bars=False)
    assert s.shape == (500, 4, 2) and torch.all(s.abs() <= 1.0)
    members = ensemble_members(post)
    assert len(members) == 2 and members[0].posterior_estimator is ens.members[0]


def test_ensemble_requires_matching_members() -> None:
    with pytest.raises(ValueError):
        FlowEnsemble([])
    with pytest.raises(ValueError):
        FlowEnsemble([ShiftedGaussian(0.0, dim=2), ShiftedGaussian(0.0, dim=3)])
    with pytest.raises(ValueError):
        TrainingConfig(ensemble_size=0)


def _toy_data(n: int = 256) -> tuple[ParameterSpace, np.ndarray, np.ndarray]:
    space = ParameterSpace()
    rng = np.random.default_rng(0)
    theta = space.sample_prior(n, rng)
    x = np.abs(rng.standard_normal((n, N_FEATURES))) + theta[:, :1]
    return space, theta, x


def test_trained_members_do_not_depend_on_parallelism() -> None:
    space, theta, x = _toy_data()
    cfg = dataclasses.replace(TINY.training, max_num_epochs=2, ensemble_size=2)
    serial, info_s = train_ensemble(theta, x, space, cfg, seeds=(11, 12), workers=1)
    parallel, info_p = train_ensemble(theta, x, space, cfg, seeds=(11, 12), workers=2)
    assert [i["validation_loss"] for i in info_s] == [i["validation_loss"] for i in info_p]
    for a, b in zip(ensemble_members(serial), ensemble_members(parallel), strict=True):
        sa, sb = a.posterior_estimator.state_dict(), b.posterior_estimator.state_dict()
        assert all(torch.equal(sa[key], sb[key]) for key in sa)
    # Different seeds give different members.
    m0, m1 = (m.posterior_estimator.state_dict() for m in ensemble_members(serial))
    assert not all(torch.equal(m0[key], m1[key]) for key in m0)
    assert isinstance(serial.posterior_estimator, FlowEnsemble)


def test_legacy_single_flow_file_loads(tmp_path: Path) -> None:
    space, theta, x = _toy_data()
    cfg = dataclasses.replace(TINY.training, max_num_epochs=2, ensemble_size=1)
    posterior, _ = train_npe(theta, x, space, cfg, seed=0)
    training = asdict(cfg)
    del training["ensemble_size"]  # written before ensembles existed
    torch.save(
        {
            "state_dict": posterior.posterior_estimator.state_dict(),
            "inferred": list(space.names),
            "training": training,
            "simulation": asdict(TINY.simulation),
            "features": list(FEATURE_NAMES),
        },
        tmp_path / "old.pt",
    )
    loaded = load_posterior(tmp_path / "old.pt")
    assert loaded.training.ensemble_size == 1
    assert not isinstance(loaded.posterior.posterior_estimator, FlowEnsemble)
    th = theta[None, :8]
    np.testing.assert_allclose(
        log_prob_unnormalized(posterior, th, x[:8]),
        log_prob_unnormalized(loaded.posterior, th, x[:8]),
        rtol=1e-5,
    )
    assert math.isfinite(float(log_prob_unnormalized(loaded.posterior, th, x[:8]).sum()))


def test_rejection_sampler_returns_the_truncated_distribution() -> None:
    torch.manual_seed(2)
    post = DirectPosterior(
        posterior_estimator=ShiftedGaussian(0.0), prior=BoxUniform(-torch.ones(2), torch.ones(2))
    )
    # The second observation keeps only ~5e-4 of its proposals.
    s = sample_batched(post, np.array([[0.0, 0.0], [3.0, 3.0]]), 4000)
    assert s.shape == (4000, 2, 2) and np.all(np.abs(s) <= 1.0)
    for j, loc in enumerate((0.0, 3.0)):
        exact = stats.truncnorm(-1 - loc, 1 - loc, loc=loc)
        np.testing.assert_allclose(s[:, j].mean(axis=0), exact.mean(), atol=4 * exact.std() / np.sqrt(4000))
        np.testing.assert_allclose(s[:, j].std(axis=0), exact.std(), rtol=0.05)
    single = sample(post, np.array([0.5, -0.5]), 100)
    assert single.shape == (100, 2)


def test_rejection_sampler_gives_up_on_posteriors_outside_the_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "MAX_PROPOSALS_PER_OBSERVATION", 50_000)
    post = DirectPosterior(
        posterior_estimator=ShiftedGaussian(0.0), prior=BoxUniform(-torch.ones(2), torch.ones(2))
    )
    with pytest.raises(RuntimeError, match="Observation 1"):
        sample_batched(post, np.array([[0.0, 0.0], [30.0, 30.0]]), 10)
