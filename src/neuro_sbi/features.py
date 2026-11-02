"""Electrophysiological summary statistics of a step-current response.

The feature set follows standard practice in single-cell electrophysiology
(e.g. the Allen Cell Types feature extractor) and the SBI literature
(Goncalves et al. 2020). Spike-shape features are measured on the *first*
action potential evoked by the stimulus.

Features that are undefined for a given response (e.g. AP width when the
cell does not fire) are replaced by documented, deterministic sentinel values
instead of being dropped. Discarding such simulations would silently truncate
the prior and bias the posterior; keeping them with sentinels lets the
density estimator learn that, e.g., "zero spikes" is itself informative. The
spike count always tells the estimator which features are sentinels.

==================  =====  ==================================================
feature             unit   definition (sentinel if undefined)
==================  =====  ==================================================
spike_count         --     APs (peaks > -10 mV, prominence > 20 mV) during the
                           stimulus
resting_mean        mV     mean V before stimulus onset
resting_std         mV     standard deviation of V before stimulus onset
stim_mean           mV     mean V during the stimulus
stim_std            mV     standard deviation of V during the stimulus
latency             ms     first AP peak time minus stimulus onset
                           (censored: stimulus duration)
firing_rate         Hz     (n - 1) / (t_last - t_first), n >= 2 (0)
ap_threshold        mV     V where dV/dt first exceeds 10 mV/ms before the
                           first AP (censored: max V during stimulus)
ap_amplitude        mV     first AP peak minus AP threshold (0)
ap_width            ms     full width at half amplitude of the first AP (0)
ahp_depth           mV     AP threshold minus the minimum V after the first AP
                           and before the next AP / within 25 ms (0)
adaptation_index    --     mean (ISI_{k+1} - ISI_k)/(ISI_{k+1} + ISI_k),
                           n >= 3 (0)
==================  =====  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from neuro_sbi.simulator import SimulationSettings

FloatArray = NDArray[np.float64]

FEATURE_NAMES: tuple[str, ...] = (
    "spike_count",
    "resting_mean",
    "resting_std",
    "stim_mean",
    "stim_std",
    "latency",
    "firing_rate",
    "ap_threshold",
    "ap_amplitude",
    "ap_width",
    "ahp_depth",
    "adaptation_index",
)

FEATURE_UNITS: dict[str, str] = {
    "spike_count": "",
    "resting_mean": "mV",
    "resting_std": "mV",
    "stim_mean": "mV",
    "stim_std": "mV",
    "latency": "ms",
    "firing_rate": "Hz",
    "ap_threshold": "mV",
    "ap_amplitude": "mV",
    "ap_width": "ms",
    "ahp_depth": "mV",
    "adaptation_index": "",
}

N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class FeatureSettings:
    """Thresholds used by the feature extractor."""

    peak_height: float = -10.0  # minimum AP peak voltage [mV]
    peak_prominence: float = 20.0  # minimum AP prominence [mV]
    min_isi: float = 1.0  # refractory distance between detected peaks [ms]
    dvdt_threshold: float = 10.0  # AP onset criterion [mV/ms]
    onset_search: float = 5.0  # look-back window for the AP onset [ms]
    ahp_window: float = 25.0  # maximal AHP search window after the peak [ms]


def detect_spikes(v: FloatArray, dt: float, fs: FeatureSettings = FeatureSettings()) -> NDArray[np.intp]:
    """Indices of AP peaks in a single voltage trace."""
    peaks, _ = find_peaks(
        v,
        height=fs.peak_height,
        prominence=fs.peak_prominence,
        distance=max(1, int(round(fs.min_isi / dt))),
    )
    return np.asarray(peaks, dtype=np.intp)


def _interp_crossing(v: FloatArray, i0: int, level: float, dt: float) -> float:
    """Time (in samples * dt) where the segment ``v[i0] -> v[i0 + 1]`` crosses
    ``level`` (linear interpolation)."""
    dv = v[i0 + 1] - v[i0]
    frac = 0.5 if dv == 0 else (level - v[i0]) / dv
    return (i0 + float(np.clip(frac, 0.0, 1.0))) * dt


def trace_features(
    v: FloatArray, settings: SimulationSettings, fs: FeatureSettings = FeatureSettings()
) -> FloatArray:
    """Compute the summary-statistic vector of one voltage trace."""
    dt = settings.dt
    i_on, i_off = settings.n_pre, settings.n_pre + settings.n_stim
    rest, stim = v[:i_on], v[i_on:i_off]
    feats = np.zeros(N_FEATURES, dtype=np.float64)
    feats[1] = rest.mean()
    feats[2] = rest.std()
    feats[3] = stim.mean()
    feats[4] = stim.std()

    peaks = detect_spikes(v, dt, fs)
    peaks = peaks[(peaks >= i_on) & (peaks < i_off)]
    n = peaks.size
    feats[0] = n
    if n == 0:
        feats[5] = settings.t_stim  # latency censored at the stimulus duration
        feats[7] = np.max(stim)  # threshold censored: the cell did not fire below this
        return feats

    t_peaks = peaks * dt
    feats[5] = t_peaks[0] - settings.t_on
    if n >= 2:
        feats[6] = 1000.0 * (n - 1) / (t_peaks[-1] - t_peaks[0])
    if n >= 3:
        isi = np.diff(t_peaks)
        feats[11] = float(np.mean((isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])))

    # First-AP shape.
    ip = int(peaks[0])
    lo = max(0, ip - int(round(fs.onset_search / dt)))
    # Onset: last sub-threshold slope before the steepest point of the upstroke.
    dvdt = np.diff(v[lo : ip + 1]) / dt
    i_max = int(np.argmax(dvdt))
    below = np.flatnonzero(dvdt[: i_max + 1] < fs.dvdt_threshold)
    io = lo + (int(below[-1]) + 1 if below.size else 0)
    v_th, v_pk = v[io], v[ip]
    amp = v_pk - v_th
    feats[7] = v_th
    feats[8] = amp

    half = v_th + 0.5 * amp
    rise = np.flatnonzero(v[io : ip + 1] < half)
    t_rise = _interp_crossing(v, io + int(rise[-1]), half, dt) if rise.size else io * dt
    stop = int(peaks[1]) if n >= 2 else min(v.size - 1, ip + int(round(fs.ahp_window / dt)))
    fall = np.flatnonzero(v[ip : stop + 1] < half)
    t_fall = _interp_crossing(v, ip + int(fall[0]) - 1, half, dt) if fall.size else stop * dt
    feats[9] = t_fall - t_rise
    feats[10] = v_th - np.min(v[ip : stop + 1])
    return feats


def compute_features(
    v: FloatArray, settings: SimulationSettings, fs: FeatureSettings = FeatureSettings()
) -> FloatArray:
    """Summary statistics for a batch of traces, shape ``(n, N_FEATURES)``."""
    v = np.atleast_2d(v)
    out = np.empty((v.shape[0], N_FEATURES), dtype=np.float64)
    for i in range(v.shape[0]):
        out[i] = trace_features(v[i], settings, fs)
    return out
