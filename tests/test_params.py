from __future__ import annotations

import numpy as np
import pytest
import torch

from neuro_sbi.params import DEFAULT_INFERRED, PARAMETER_TABLE, ParameterSpace, ParameterSpec


def test_default_space_and_prior() -> None:
    space = ParameterSpace()
    assert space.names == DEFAULT_INFERRED
    prior = space.prior()
    s = prior.sample((1000,))
    b = torch.as_tensor(space.bounds, dtype=torch.float32)
    assert torch.all(s >= b[:, 0]) and torch.all(s <= b[:, 1])
    assert torch.isfinite(prior.log_prob(torch.as_tensor(space.ground_truth(), dtype=torch.float32)))


def test_log_scale_roundtrip_and_ground_truth() -> None:
    space = ParameterSpace()
    gt = space.ground_truth()
    phys = space.physical_table(gt[None])[0]
    expected = [PARAMETER_TABLE[n].default for n in space.names]
    np.testing.assert_allclose(phys, expected)
    for j, spec in enumerate(space.specs):
        np.testing.assert_allclose(spec.to_inference(phys[j]), gt[j])
        lo, hi = space.bounds[j]
        assert lo < gt[j] < hi


def test_fixed_parameters_are_broadcast() -> None:
    space = ParameterSpace(("g_Na", "g_K"))
    theta = space.sample_prior(5, np.random.default_rng(0))
    p = space.to_physical(theta)
    np.testing.assert_allclose(p["g_leak"], PARAMETER_TABLE["g_leak"].default)
    np.testing.assert_allclose(np.log10(p["g_Na"]), theta[:, 0])
    assert all(v.shape == (5,) for v in p.values())


def test_invalid_specs() -> None:
    with pytest.raises(KeyError):
        ParameterSpace(("g_Ca",))
    with pytest.raises(ValueError):
        ParameterSpace(("g_Na", "g_Na"))
    with pytest.raises(ValueError):
        ParameterSpec("x", "x", "mV", -1.0, 1.0, True, 0.5)
    with pytest.raises(ValueError):
        ParameterSpec("x", "x", "mV", 0.0, 1.0, False, 2.0)
