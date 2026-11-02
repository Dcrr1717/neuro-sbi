"""Pipeline configurations.

* ``tiny``  -- seconds; used by the unit tests (short protocol, few simulations).
* ``small`` -- default; a few minutes on a laptop/CI runner. Exercises every
  stage of the pipeline end-to-end but with a budget too small for
  publication-quality posteriors.
* ``full``  -- the real run whose results are committed under ``results/``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from neuro_sbi.params import DEFAULT_INFERRED
from neuro_sbi.simulator import SimulationSettings


@dataclass(frozen=True)
class TrainingConfig:
    """Neural posterior estimator and optimizer settings."""

    density_estimator: str = "nsf"  # "nsf" (neural spline flow) or "maf"
    hidden_features: int = 64
    num_transforms: int = 5
    num_bins: int = 10  # spline bins (NSF only)
    training_batch_size: int = 256
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    max_num_epochs: int = 1000


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Budgets for the validation stage."""

    num_sbc: int = 1000  # independent (theta, x) pairs for SBC / coverage / recovery
    sbc_samples: int = 1000  # posterior draws per SBC pair (L)
    sbc_bins: int = 20  # histogram bins for the rank plots
    num_posterior_samples: int = 10_000  # draws for the worked example
    num_ppc: int = 500  # posterior- and prior-predictive simulations
    num_ppc_traces: int = 4  # traces shown in the PPC figure


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    num_simulations: int
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    inferred: tuple[str, ...] = DEFAULT_INFERRED
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    seed: int = 2020
    chunk_size: int = 500  # neurons simulated in lock-step per worker task
    output_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


TINY = PipelineConfig(
    name="tiny",
    num_simulations=300,
    simulation=SimulationSettings(dt=0.05, t_burn=10.0, t_pre=10.0, t_stim=40.0, t_post=10.0),
    training=TrainingConfig(
        density_estimator="maf",
        hidden_features=16,
        num_transforms=2,
        training_batch_size=64,
        learning_rate=1e-3,
        stop_after_epochs=3,
        max_num_epochs=5,
    ),
    diagnostics=DiagnosticsConfig(
        num_sbc=30,
        sbc_samples=50,
        sbc_bins=5,
        num_posterior_samples=200,
        num_ppc=10,
        num_ppc_traces=2,
    ),
    chunk_size=100,
    output_dir="results/tiny",
)

SMALL = PipelineConfig(
    name="small",
    num_simulations=4_000,
    training=TrainingConfig(
        density_estimator="maf",
        hidden_features=50,
        num_transforms=3,
        training_batch_size=128,
        learning_rate=1e-3,
        stop_after_epochs=15,
        max_num_epochs=300,
    ),
    diagnostics=DiagnosticsConfig(
        num_sbc=200,
        sbc_samples=200,
        sbc_bins=10,
        num_posterior_samples=5_000,
        num_ppc=100,
    ),
    output_dir="results/small",
)

FULL = PipelineConfig(
    name="full",
    num_simulations=100_000,
    training=TrainingConfig(
        density_estimator="nsf",
        hidden_features=64,
        num_transforms=5,
        training_batch_size=512,
        learning_rate=5e-4,
        stop_after_epochs=20,
        max_num_epochs=1000,
    ),
    diagnostics=DiagnosticsConfig(
        num_sbc=1000,
        sbc_samples=1000,
        sbc_bins=20,
        num_posterior_samples=20_000,
        num_ppc=500,
    ),
    output_dir="results",
)

CONFIGS: dict[str, PipelineConfig] = {c.name: c for c in (TINY, SMALL, FULL)}


def get_config(name: str, **overrides: Any) -> PipelineConfig:
    """Look up a named configuration, optionally overriding top-level fields."""
    try:
        cfg = CONFIGS[name]
    except KeyError as err:
        raise KeyError(f"Unknown config {name!r}; choose from {sorted(CONFIGS)}") from err
    return dataclasses.replace(cfg, **overrides) if overrides else cfg
