"""Tests of plotting helpers that compute numbers (not of the figures' looks)."""

from __future__ import annotations

import numpy as np

from neuro_sbi.plotting import posterior_limits


def test_posterior_limits_cover_samples_and_truth_inside_prior() -> None:
    rng = np.random.default_rng(0)
    bounds = np.array([[0.0, 10.0], [-5.0, 5.0], [0.0, 1.0]])
    samples = np.column_stack(
        [rng.normal(5.0, 0.01, 20_000), rng.normal(4.9, 0.5, 20_000), rng.uniform(0, 1, 20_000)]
    )
    truth = np.array([5.2, 0.0, 0.5])
    lim = posterior_limits(samples, truth, bounds)
    assert np.all(lim[:, 0] >= bounds[:, 0]) and np.all(lim[:, 1] <= bounds[:, 1])
    assert np.all((lim[:, 0] < truth) & (truth < lim[:, 1]))
    # A narrow posterior is zoomed in; one hitting the prior edge is clipped there.
    assert lim[0, 1] - lim[0, 0] < 0.05 * (bounds[0, 1] - bounds[0, 0])
    assert lim[1, 1] == bounds[1, 1]
    np.testing.assert_allclose(lim[2], bounds[2])
