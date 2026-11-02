"""Single-compartment Hodgkin-Huxley neuron driven by a step current.

The model is the cortical point neuron of Pospischil et al. (2008), which is
also the model used by Goncalves et al. (2020, eLife) to introduce neural
posterior estimation to neuroscience. The membrane equation is

    C dV/dt = -g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_M p (V - E_K)
              - g_leak (V - E_leak) + I_inj(t) + C * sigma * xi(t),

with Hodgkin-Huxley-type gating for m, h, n (shifted by the threshold
parameter V_T), a slow non-inactivating M-type K+ current p (spike-frequency
adaptation) and white membrane noise of intensity ``sigma`` (mV/sqrt(ms)).

Units: V [mV], t [ms], g [mS/cm²], I [µA/cm²], C [µF/cm²].

Two integrators are provided:

* :func:`simulate_batch` -- vectorized *exponential Euler* scheme (Rush-Larsen
  update for the gates, exponential update for V given the gates,
  Euler-Maruyama for the noise). It simulates thousands of neurons in
  lock-step with numpy and is unconditionally stable for the stiff sodium
  dynamics, which makes it suitable for generating SBI training data.
* :func:`simulate_scipy` -- adaptive-step reference solution of the
  deterministic ODE (``sigma = 0``) with :func:`scipy.integrate.solve_ivp`
  (LSODA). Used in the test-suite to verify the fast integrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationSettings:
    """Stimulation protocol, numerics and fixed biophysical constants.

    The default step current of 3 µA/cm² corresponds to ~0.46 nA injected into
    a 70 µm x 70 µm cylindrical soma (area 1.54e-4 cm²).
    """

    dt: float = 0.025  # integration step [ms]
    t_burn: float = 50.0  # discarded burn-in after steady-state initialization [ms]
    t_pre: float = 20.0  # recorded baseline before stimulus onset [ms]
    t_stim: float = 100.0  # duration of the current step [ms]
    t_post: float = 30.0  # recorded period after stimulus offset [ms]
    i_amp: float = 3.0  # step amplitude [µA/cm²]
    noise_std: float = 0.1  # membrane noise intensity [mV/sqrt(ms)]
    c_m: float = 1.0  # membrane capacitance [µF/cm²]
    e_na: float = 53.0  # Na+ reversal potential [mV]
    e_k: float = -107.0  # K+ reversal potential [mV]

    def __post_init__(self) -> None:
        for name in ("dt", "t_pre", "t_stim", "c_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("t_burn", "t_post", "noise_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def _steps(self, duration: float) -> int:
        return int(round(duration / self.dt))

    @property
    def n_burn(self) -> int:
        return self._steps(self.t_burn)

    @property
    def n_pre(self) -> int:
        return self._steps(self.t_pre)

    @property
    def n_stim(self) -> int:
        return self._steps(self.t_stim)

    @property
    def n_post(self) -> int:
        return self._steps(self.t_post)

    @property
    def n_record(self) -> int:
        """Number of recorded samples."""
        return self.n_pre + self.n_stim + self.n_post

    @property
    def t_on(self) -> float:
        """Stimulus onset, relative to the start of the recording [ms]."""
        return self.n_pre * self.dt

    @property
    def t_off(self) -> float:
        """Stimulus offset, relative to the start of the recording [ms]."""
        return (self.n_pre + self.n_stim) * self.dt

    def time(self) -> FloatArray:
        """Recording time axis [ms] (0 = start of recording)."""
        return np.arange(self.n_record, dtype=np.float64) * self.dt

    def stimulus(self) -> FloatArray:
        """Injected current on the recording time axis [µA/cm²]."""
        i = np.zeros(self.n_record, dtype=np.float64)
        i[self.n_pre : self.n_pre + self.n_stim] = self.i_amp
        return i


# --------------------------------------------------------------------------
# Channel kinetics (Pospischil et al. 2008, Traub & Miles formalism)
# --------------------------------------------------------------------------


def _vtrap(x: FloatArray, y: float) -> FloatArray:
    """Numerically safe ``x / (exp(x / y) - 1)`` (limit ``y`` at ``x = 0``)."""
    z = x / y
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return y * z / np.expm1(z)


def gating_rates(
    v: FloatArray, v_t: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Opening/closing rates [1/ms] of the m, h and n gates."""
    u = v - v_t
    a_m = 0.32 * _vtrap(13.0 - u, 4.0)
    b_m = 0.28 * _vtrap(u - 40.0, 5.0)
    a_h = 0.128 * np.exp((17.0 - u) / 18.0)
    b_h = 4.0 / (1.0 + np.exp((40.0 - u) / 5.0))
    a_n = 0.032 * _vtrap(15.0 - u, 5.0)
    b_n = 0.5 * np.exp((10.0 - u) / 40.0)
    return a_m, b_m, a_h, b_h, a_n, b_n


def m_current_kinetics(v: FloatArray, tau_max: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Steady state and time constant [ms] of the slow M-current gate p."""
    p_inf = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    tau_p = tau_max / (3.3 * np.exp((v + 35.0) / 20.0) + np.exp(-(v + 35.0) / 20.0))
    return p_inf, tau_p


def steady_state_gates(
    v: FloatArray, v_t: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Steady-state values of (m, h, n, p) at voltage ``v``."""
    a_m, b_m, a_h, b_h, a_n, b_n = gating_rates(v, v_t)
    p_inf, _ = m_current_kinetics(v, np.ones_like(v))
    return a_m / (a_m + b_m), a_h / (a_h + b_h), a_n / (a_n + b_n), p_inf


def _broadcast_params(params: Mapping[str, FloatArray | float]) -> dict[str, FloatArray]:
    required = ("g_Na", "g_K", "g_leak", "g_M", "E_leak", "V_T", "tau_max")
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"Missing simulator parameters: {missing}")
    arrays = {k: np.atleast_1d(np.asarray(params[k], dtype=np.float64)) for k in required}
    shape = np.broadcast_shapes(*(a.shape for a in arrays.values()))
    if len(shape) != 1:
        raise ValueError("Parameters must be scalars or 1-D arrays")
    out = {k: np.broadcast_to(a, shape).copy() for k, a in arrays.items()}
    for k in ("g_Na", "g_K", "g_leak", "g_M", "tau_max"):
        if np.any(out[k] < 0) or (k in ("g_leak", "tau_max") and np.any(out[k] <= 0)):
            raise ValueError(f"{k} must be positive")
    return out


def resting_potential(
    params: Mapping[str, FloatArray | float], settings: SimulationSettings, n_iter: int = 80
) -> FloatArray:
    """Zero-current voltage with all gates at steady state (vectorized bisection).

    The total steady-state current is negative at -120 mV and positive at
    0 mV for every parameter set in the prior, so a root exists in between.
    If the cell has no stable resting state (tonic firing) the returned value
    is merely a well-defined initial condition.
    """
    p = _broadcast_params(params)
    n = p["g_Na"].shape[0]
    lo = np.full(n, -120.0)
    hi = np.full(n, 0.0)

    def total_current(v: FloatArray) -> FloatArray:
        m, h, nn, pp = steady_state_gates(v, p["V_T"])
        return (
            p["g_Na"] * m**3 * h * (v - settings.e_na)
            + p["g_K"] * nn**4 * (v - settings.e_k)
            + p["g_M"] * pp * (v - settings.e_k)
            + p["g_leak"] * (v - p["E_leak"])
        )

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        positive = total_current(mid) > 0
        hi = np.where(positive, mid, hi)
        lo = np.where(positive, lo, mid)
    return np.asarray(0.5 * (lo + hi), dtype=np.float64)


def simulate_batch(
    params: Mapping[str, FloatArray | float],
    settings: SimulationSettings,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Simulate a batch of neurons with the exponential-Euler scheme.

    Args:
        params: Physical parameters (scalars or arrays of shape ``(n,)``) with
            keys ``g_Na, g_K, g_leak, g_M, E_leak, V_T, tau_max``.
        settings: Protocol and numerical settings.
        rng: Random generator for the membrane noise. Required if
            ``settings.noise_std > 0``.

    Returns:
        Membrane voltage of shape ``(n, settings.n_record)`` [mV].
    """
    p = _broadcast_params(params)
    if settings.noise_std > 0 and rng is None:
        raise ValueError("An rng is required when noise_std > 0")
    n = p["g_Na"].shape[0]
    dt = settings.dt
    g_na, g_k, g_l, g_m = p["g_Na"], p["g_K"], p["g_leak"], p["g_M"]
    e_l, v_t, tau_max = p["E_leak"], p["V_T"], p["tau_max"]
    e_na, e_k, c_m = settings.e_na, settings.e_k, settings.c_m

    v = resting_potential(p, settings)
    m, h, n_gate, p_gate = steady_state_gates(v, v_t)

    n_total = settings.n_burn + settings.n_record
    stim_start = settings.n_burn + settings.n_pre
    stim_stop = stim_start + settings.n_stim
    noise_scale = settings.noise_std * np.sqrt(dt)
    out = np.empty((n, settings.n_record), dtype=np.float64)

    for k in range(n_total):
        if k >= settings.n_burn:
            out[:, k - settings.n_burn] = v
        i_inj = settings.i_amp if stim_start <= k < stim_stop else 0.0

        # Gates: Rush-Larsen (exact for frozen V).
        a_m, b_m, a_h, b_h, a_n, b_n = gating_rates(v, v_t)
        s_m, s_h, s_n = a_m + b_m, a_h + b_h, a_n + b_n
        m = a_m / s_m + (m - a_m / s_m) * np.exp(-dt * s_m)
        h = a_h / s_h + (h - a_h / s_h) * np.exp(-dt * s_h)
        n_gate = a_n / s_n + (n_gate - a_n / s_n) * np.exp(-dt * s_n)
        p_inf, tau_p = m_current_kinetics(v, tau_max)
        p_gate = p_inf + (p_gate - p_inf) * np.exp(-dt / tau_p)

        # Voltage: exponential update towards the instantaneous steady state.
        g_na_eff = g_na * m**3 * h
        g_k_eff = g_k * n_gate**4 + g_m * p_gate
        g_tot = g_na_eff + g_k_eff + g_l
        v_inf = (g_na_eff * e_na + g_k_eff * e_k + g_l * e_l + i_inj) / g_tot
        v = v_inf + (v - v_inf) * np.exp(-dt * g_tot / c_m)
        if noise_scale > 0 and rng is not None:
            v = v + noise_scale * rng.standard_normal(n)
    return out


def simulate_scipy(
    params: Mapping[str, float],
    settings: SimulationSettings,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> FloatArray:
    """Deterministic reference solution with :func:`scipy.integrate.solve_ivp`.

    Membrane noise is ignored (``sigma = 0``). The integration is restarted
    at the discontinuities of the step current. Returns the voltage of a
    single neuron on the recording time axis, shape ``(settings.n_record,)``.
    """
    p = {k: float(np.asarray(v).reshape(())) for k, v in _broadcast_params(params).items()}

    def rhs(_t: float, y: FloatArray, i_inj: float) -> FloatArray:
        v, m, h, n_gate, p_gate = y
        va = np.array([v])
        a_m, b_m, a_h, b_h, a_n, b_n = gating_rates(va, np.array([p["V_T"]]))
        p_inf, tau_p = m_current_kinetics(va, np.array([p["tau_max"]]))
        i_ion = (
            p["g_Na"] * m**3 * h * (v - settings.e_na)
            + p["g_K"] * n_gate**4 * (v - settings.e_k)
            + p["g_M"] * p_gate * (v - settings.e_k)
            + p["g_leak"] * (v - p["E_leak"])
        )
        return np.array(
            [
                (i_inj - i_ion) / settings.c_m,
                a_m[0] * (1 - m) - b_m[0] * m,
                a_h[0] * (1 - h) - b_h[0] * h,
                a_n[0] * (1 - n_gate) - b_n[0] * n_gate,
                (p_inf[0] - p_gate) / tau_p[0],
            ]
        )

    v0 = resting_potential(p, settings)
    gates = steady_state_gates(v0, np.array([p["V_T"]]))
    y0 = np.array([v0[0], *(g[0] for g in gates)])

    dt = settings.dt
    t_rec0 = settings.n_burn * dt
    t_grid = t_rec0 + settings.time()
    t_on_abs = t_rec0 + settings.t_on
    t_off_abs = t_rec0 + settings.t_off
    t_end = t_rec0 + settings.n_record * dt
    segments = ((0.0, t_on_abs, 0.0), (t_on_abs, t_off_abs, settings.i_amp), (t_off_abs, t_end, 0.0))

    v_out = np.empty(settings.n_record, dtype=np.float64)
    y = y0
    for t0, t1, current in segments:
        sel = (t_grid >= t0 - 1e-9) & (t_grid < t1 - 1e-9)
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y,
            method="LSODA",
            t_eval=t_grid[sel],
            args=(current,),
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        v_out[sel] = sol.y[0]
        y = sol.sol(t1)
    return v_out
