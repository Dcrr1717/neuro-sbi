"""Chunked, reproducible, optionally parallel simulation of many neurons.

Each chunk of ``chunk_size`` neurons gets its own random stream derived from
``(seed, chunk_index)`` via :class:`numpy.random.SeedSequence`, so results are
bit-identical regardless of the number of worker processes.

This module deliberately imports only numpy/scipy so worker processes start
quickly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from numpy.typing import NDArray

from neuro_sbi.features import FeatureSettings, compute_features
from neuro_sbi.simulator import SimulationSettings, simulate_batch

FloatArray = NDArray[np.float64]


def _chunk_rng(seed: int, chunk: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(chunk,)))


def _run_chunk(
    args: tuple[dict[str, FloatArray], SimulationSettings, FeatureSettings, int, int, bool],
) -> tuple[FloatArray, FloatArray | None]:
    params, settings, fs, seed, chunk, keep_traces = args
    v = simulate_batch(params, settings, _chunk_rng(seed, chunk))
    return compute_features(v, settings, fs), (v if keep_traces else None)


def default_workers() -> int:
    """Number of worker processes (``NEURO_SBI_WORKERS`` or the CPU count)."""
    env = os.environ.get("NEURO_SBI_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def simulate(
    params: Mapping[str, FloatArray],
    settings: SimulationSettings,
    seed: int,
    chunk_size: int = 500,
    workers: int = 1,
    keep_traces: bool = False,
    feature_settings: FeatureSettings = FeatureSettings(),
) -> tuple[FloatArray, FloatArray | None]:
    """Simulate neurons and compute their summary statistics.

    Args:
        params: Physical parameters, each an array of shape ``(n,)``.
        settings: Simulation protocol.
        seed: Seed of this simulation campaign.
        chunk_size: Neurons per chunk (memory ~ chunk_size * n_record * 8 B).
        workers: Worker processes; 1 runs in-process.
        keep_traces: Also return the voltage traces (memory heavy).

    Returns:
        ``(features, traces)`` with features of shape ``(n, N_FEATURES)`` and
        traces of shape ``(n, n_record)`` or ``None``.
    """
    n = int(np.asarray(next(iter(params.values()))).shape[0])
    starts = list(range(0, n, chunk_size))
    tasks = [
        (
            {k: np.asarray(v)[s : s + chunk_size] for k, v in params.items()},
            settings,
            feature_settings,
            seed,
            i,
            keep_traces,
        )
        for i, s in enumerate(starts)
    ]
    if workers <= 1 or len(tasks) == 1:
        results = [_run_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=get_context("spawn")
        ) as pool:
            results = list(pool.map(_run_chunk, tasks))
    feats = np.concatenate([r[0] for r in results], axis=0)
    traces: FloatArray | None = None
    if keep_traces:
        traces = np.concatenate([r[1] for r in results if r[1] is not None], axis=0)
    return feats, traces
