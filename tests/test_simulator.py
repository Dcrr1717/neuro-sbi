from __future__ import annotations

import numpy as np
import pytest

from neuro_sbi.batch import simulate
from neuro_sbi.features import compute_features, detect_spikes
from neuro_sbi.simulator import (
    SimulationSettings,
    _vtrap,
    resting_potential,
    simulate_batch,
    simulate_scipy,
)

REFERENCE = {
    "g_Na": 50.0,
    "g_K": 5.0,
    "g_leak": 0.1,
    "g_M": 0.07,
    "E_leak": -70.0,
    "V_T": -60.0,
    "tau_max": 600.0,
}
SHORT = SimulationSettings(t_burn=10.0, t_pre=10.0, t_stim=50.0, t_post=10.0, noise_std=0.0)


def test_vtrap_is_continuous_at_singularity() -> None:
    x = np.array([-1e-7, 0.0, 1e-7])
    np.testing.assert_allclose(_vtrap(x, 4.0), 4.0, rtol=1e-6)
    np.testing.assert_allclose(_vtrap(np.array([2.0]), 4.0), 2.0 / np.expm1(0.5))


def test_exponential_euler_matches_scipy_reference() -> None:
    """The fast integrator reproduces an adaptive LSODA solution."""
    v_fast = simulate_batch(REFERENCE, SHORT)[0]
    v_ref = simulate_scipy(REFERENCE, SHORT)
    p_fast = detect_spikes(v_fast, SHORT.dt)
    p_ref = detect_spikes(v_ref, SHORT.dt)
    assert p_fast.size == p_ref.size >= 2
    np.testing.assert_allclose(p_fast * SHORT.dt, p_ref * SHORT.dt, atol=0.1)  # ms
    np.testing.assert_allclose(v_fast[: SHORT.n_pre], v_ref[: SHORT.n_pre], atol=1e-3)
    f_fast = compute_features(v_fast, SHORT)[0]
    f_ref = compute_features(v_ref, SHORT)[0]
    np.testing.assert_allclose(f_fast, f_ref, rtol=0.02, atol=0.05)


def test_resting_state_is_a_fixed_point() -> None:
    s = SimulationSettings(t_burn=0.0, t_pre=20.0, t_stim=1.0, t_post=0.0, i_amp=0.0, noise_std=0.0)
    params: dict[str, np.ndarray | float] = {
        **REFERENCE,
        "g_leak": np.array([0.05, 0.1, 0.3]),
        "E_leak": np.array([-80.0, -70.0, -60.0]),
    }
    v = simulate_batch(params, s)
    v0 = resting_potential(params, s)
    np.testing.assert_allclose(v, np.repeat(v0[:, None], v.shape[1], axis=1), atol=1e-8)


def test_noise_is_reproducible_and_seed_dependent() -> None:
    s = SimulationSettings(t_burn=5.0, t_pre=10.0, t_stim=10.0, t_post=5.0)
    a = simulate_batch(REFERENCE, s, np.random.default_rng(1))
    b = simulate_batch(REFERENCE, s, np.random.default_rng(1))
    c = simulate_batch(REFERENCE, s, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    with pytest.raises(ValueError):
        simulate_batch(REFERENCE, s)  # noise without rng


def test_stimulus_evokes_spikes_and_more_current_more_spikes() -> None:
    counts = []
    for amp in (0.0, 3.0, 8.0):
        s = SimulationSettings(t_burn=10.0, t_pre=10.0, t_stim=100.0, t_post=10.0, i_amp=amp, noise_std=0.0)
        counts.append(compute_features(simulate_batch(REFERENCE, s), s)[0, 0])
    assert counts[0] == 0
    assert 0 < counts[1] < counts[2]


def test_parallel_simulation_is_bit_identical() -> None:
    s = SimulationSettings(dt=0.05, t_burn=5.0, t_pre=10.0, t_stim=30.0, t_post=5.0)
    rng = np.random.default_rng(0)
    params = {k: np.full(6, v) for k, v in REFERENCE.items()}
    params["g_Na"] = rng.uniform(20, 80, 6)
    f1, _ = simulate(params, s, seed=7, chunk_size=2, workers=1)
    f2, _ = simulate(params, s, seed=7, chunk_size=2, workers=2)
    np.testing.assert_array_equal(f1, f2)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        simulate_batch({**REFERENCE, "g_leak": -0.1}, SHORT)
    with pytest.raises(KeyError):
        simulate_batch({"g_Na": 1.0}, SHORT)
    with pytest.raises(ValueError):
        SimulationSettings(dt=0.0)
