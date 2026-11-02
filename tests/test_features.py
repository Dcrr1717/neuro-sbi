from __future__ import annotations

import numpy as np

from neuro_sbi.features import FEATURE_NAMES, N_FEATURES, compute_features
from neuro_sbi.simulator import SimulationSettings

S = SimulationSettings(dt=0.05, t_pre=20.0, t_stim=100.0, t_post=20.0)
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _synthetic_trace(peak_times: list[float], base: float = -65.0) -> np.ndarray:
    """Piecewise-linear APs: 0.5 ms rise to +35 mV, 1 ms fall to -75 mV,
    2 ms recovery to baseline. Breakpoints lie on the sampling grid, so
    linear interpolation in the feature extractor is exact."""
    t = S.time()
    v = np.full_like(t, base)
    for tp in peak_times:
        knots_t = [tp - 0.5, tp, tp + 1.0, tp + 3.0]
        knots_v = [base, 35.0, -75.0, base]
        seg = (t >= knots_t[0]) & (t <= knots_t[-1])
        v[seg] = np.interp(t[seg], knots_t, knots_v)
    return v


def test_feature_vector_layout() -> None:
    assert N_FEATURES == len(FEATURE_NAMES) == 12


def test_spiking_trace_features_are_exact() -> None:
    peaks = [S.t_on + 10.0, S.t_on + 30.0, S.t_on + 60.0]  # ISIs 20 and 30 ms
    f = compute_features(_synthetic_trace(peaks), S)[0]
    assert f[IDX["spike_count"]] == 3
    np.testing.assert_allclose(f[IDX["latency"]], 10.0, atol=1e-9)
    np.testing.assert_allclose(f[IDX["firing_rate"]], 1000.0 * 2 / 50.0)
    np.testing.assert_allclose(f[IDX["adaptation_index"]], (30 - 20) / (30 + 20))
    np.testing.assert_allclose(f[IDX["ap_threshold"]], -65.0)
    np.testing.assert_allclose(f[IDX["ap_amplitude"]], 100.0)
    # Half amplitude (-15 mV) is crossed 0.25 ms after onset and 50/110 ms after the peak.
    np.testing.assert_allclose(f[IDX["ap_width"]], 0.25 + 50.0 / 110.0, atol=1e-9)
    np.testing.assert_allclose(f[IDX["ahp_depth"]], 10.0)
    np.testing.assert_allclose(f[IDX["resting_mean"]], -65.0)
    np.testing.assert_allclose(f[IDX["resting_std"]], 0.0, atol=1e-12)


def test_spikes_outside_stimulus_are_not_counted() -> None:
    f = compute_features(_synthetic_trace([5.0, S.t_off + 5.0]), S)[0]
    assert f[IDX["spike_count"]] == 0


def test_silent_trace_uses_documented_sentinels() -> None:
    rng = np.random.default_rng(0)
    v = -70.0 + 0.2 * rng.standard_normal(S.n_record)
    v[S.n_pre : S.n_pre + S.n_stim] += 8.0  # subthreshold depolarization
    f = compute_features(v, S)[0]
    assert f[IDX["spike_count"]] == 0
    assert f[IDX["latency"]] == S.t_stim
    assert f[IDX["ap_threshold"]] == np.max(v[S.n_pre : S.n_pre + S.n_stim])
    for name in ("firing_rate", "ap_amplitude", "ap_width", "ahp_depth", "adaptation_index"):
        assert f[IDX[name]] == 0.0
    np.testing.assert_allclose(f[IDX["resting_mean"]], v[: S.n_pre].mean())
    np.testing.assert_allclose(f[IDX["stim_mean"]], v[S.n_pre : S.n_pre + S.n_stim].mean())


def test_small_noise_does_not_create_spikes() -> None:
    rng = np.random.default_rng(1)
    v = -20.0 + 2.0 * rng.standard_normal(S.n_record)  # depolarization block with noise
    assert compute_features(v, S)[0, IDX["spike_count"]] == 0
