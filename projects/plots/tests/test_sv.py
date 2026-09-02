import numpy as np
import pytest
from plots.legacy import compute

MASS_COMBOS = [(35, 35), (35, 20), (20, 20), (20, 10)]


@pytest.fixture
def det_stats_and_weights():
    rng = np.random.default_rng(7)
    n_events, n_combos = 400, len(MASS_COMBOS)
    det_stats = rng.exponential(2.0, size=n_events) + 5.0
    weights = rng.uniform(0, 1, size=(n_combos, n_events))
    weights /= weights.sum(axis=-1, keepdims=True)
    return det_stats, weights


def test_compute_shapes_and_monotonicity(det_stats_and_weights):
    det_stats, weights = det_stats_and_weights
    thresholds = np.sort(det_stats)[::-1][:50]

    sv, err = compute.sensitive_volume(det_stats, weights, thresholds)

    assert sv.shape == (len(MASS_COMBOS), len(thresholds))
    assert err.shape == sv.shape
    assert np.all(err >= 0)
    assert np.all(np.isfinite(sv))

    # Test monotonicity
    assert np.all(np.diff(sv, axis=-1) >= 0)


def test_compute_threshold_extremes(det_stats_and_weights):
    det_stats, weights = det_stats_and_weights
    thresholds = np.array([det_stats.max() + 1, det_stats.min() - 1])

    sv, _ = compute.sensitive_volume(det_stats, weights, thresholds)

    # nothing clears a threshold above the loudest event
    np.testing.assert_allclose(sv[:, 0], 0)
    # everything clears a threshold below the quietest
    np.testing.assert_allclose(sv[:, 1], weights.sum(axis=-1))
