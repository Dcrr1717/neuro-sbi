"""Model parameters, prior specification and parameter-space transforms.

Inference is carried out in an *inference space* in which the prior is a
box-uniform distribution. Conductances span orders of magnitude and are
strictly positive, so they are represented by ``log10(g)``: a uniform prior on
``log10(g)`` is a log-uniform (scale-invariant) prior on ``g``. Reversal
potentials and voltage offsets are represented linearly in mV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sbi.utils import BoxUniform

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParameterSpec:
    """Specification of one biophysical parameter.

    Attributes:
        name: Internal key (used in the simulator).
        symbol: Short human-readable symbol for plots and tables.
        unit: Physical unit.
        low: Lower prior bound in physical units.
        high: Upper prior bound in physical units.
        log_scale: If ``True`` the prior is uniform in ``log10`` of the value.
        default: Reference value. Used as the ground truth in the worked
            recovery and as the fixed value whenever the parameter is not
            inferred.
    """

    name: str
    symbol: str
    unit: str
    low: float
    high: float
    log_scale: bool
    default: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be < high")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"{self.name}: log-scale parameters need positive bounds")
        if not self.low <= self.default <= self.high:
            raise ValueError(f"{self.name}: default must lie inside the prior bounds")

    @property
    def inference_bounds(self) -> tuple[float, float]:
        """Prior bounds in inference space."""
        if self.log_scale:
            return float(np.log10(self.low)), float(np.log10(self.high))
        return self.low, self.high

    @property
    def axis_label(self) -> str:
        """Axis label in inference space."""
        if self.log_scale:
            return f"log10 {self.symbol} [{self.unit}]"
        return f"{self.symbol} [{self.unit}]"

    def to_physical(self, value: FloatArray) -> FloatArray:
        """Map inference-space values to physical units."""
        return np.power(10.0, value) if self.log_scale else np.asarray(value, dtype=np.float64)

    def to_inference(self, value: FloatArray | float) -> FloatArray:
        """Map physical values to inference space."""
        arr = np.asarray(value, dtype=np.float64)
        return np.log10(arr) if self.log_scale else arr


# Reference values follow the regular-spiking cortical neuron of
# Pospischil et al. (2008), as used by Goncalves et al. (2020).
PARAMETER_TABLE: dict[str, ParameterSpec] = {
    spec.name: spec
    for spec in (
        ParameterSpec("g_Na", "gNa", "mS/cm²", 5.0, 100.0, True, 50.0),
        ParameterSpec("g_K", "gK", "mS/cm²", 0.5, 20.0, True, 5.0),
        ParameterSpec("g_leak", "gleak", "mS/cm²", 0.02, 0.5, True, 0.1),
        ParameterSpec("g_M", "gM", "mS/cm²", 0.005, 0.5, True, 0.07),
        ParameterSpec("E_leak", "Eleak", "mV", -85.0, -55.0, False, -70.0),
        ParameterSpec("V_T", "VT", "mV", -70.0, -50.0, False, -60.0),
        ParameterSpec("tau_max", "τmax", "ms", 100.0, 2000.0, True, 600.0),
    )
}

DEFAULT_INFERRED: tuple[str, ...] = ("g_Na", "g_K", "g_leak", "g_M", "E_leak", "V_T")


class ParameterSpace:
    """The subset of parameters that is inferred, plus the fixed remainder."""

    def __init__(self, inferred: Sequence[str] = DEFAULT_INFERRED) -> None:
        unknown = [n for n in inferred if n not in PARAMETER_TABLE]
        if unknown:
            raise KeyError(f"Unknown parameters: {unknown}")
        if len(set(inferred)) != len(inferred):
            raise ValueError("Duplicate parameter names")
        self.names: tuple[str, ...] = tuple(inferred)
        self.specs: tuple[ParameterSpec, ...] = tuple(PARAMETER_TABLE[n] for n in inferred)

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def bounds(self) -> FloatArray:
        """Array of shape ``(dim, 2)`` with inference-space prior bounds."""
        return np.array([s.inference_bounds for s in self.specs], dtype=np.float64)

    @property
    def labels(self) -> list[str]:
        return [s.axis_label for s in self.specs]

    def prior(self) -> BoxUniform:
        """Box-uniform prior in inference space (log-uniform for conductances)."""
        b = self.bounds
        return BoxUniform(
            low=torch.as_tensor(b[:, 0], dtype=torch.float32),
            high=torch.as_tensor(b[:, 1], dtype=torch.float32),
        )

    def sample_prior(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Sample ``n`` parameter vectors (inference space) with numpy."""
        b = self.bounds
        return rng.uniform(b[:, 0], b[:, 1], size=(n, self.dim))

    def ground_truth(self, overrides: Mapping[str, float] | None = None) -> FloatArray:
        """Reference parameter vector in inference space."""
        values = {s.name: s.default for s in self.specs}
        if overrides:
            values.update(overrides)
        return np.array([float(s.to_inference(values[s.name])) for s in self.specs], dtype=np.float64)

    def to_physical(self, theta: FloatArray) -> dict[str, FloatArray]:
        """Map inference-space ``theta`` of shape ``(n, dim)`` to a full
        physical parameter dictionary (fixed parameters broadcast to ``n``)."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if theta.shape[1] != self.dim:
            raise ValueError(f"theta must have {self.dim} columns, got {theta.shape[1]}")
        n = theta.shape[0]
        out: dict[str, FloatArray] = {
            name: np.full(n, spec.default, dtype=np.float64) for name, spec in PARAMETER_TABLE.items()
        }
        for j, spec in enumerate(self.specs):
            out[spec.name] = spec.to_physical(theta[:, j])
        return out

    def physical_table(self, theta: FloatArray) -> FloatArray:
        """Inferred parameters only, in physical units, shape ``(n, dim)``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        return np.stack([s.to_physical(theta[:, j]) for j, s in enumerate(self.specs)], axis=1)
