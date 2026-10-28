import json

import numpy as np
import pytest
from plots.core import compute
from plots.core.constants import SECONDS_PER_YEAR
from plots.core.data import AnalysisData
from plots.core.sv import (
    _astrophysical_volume,
    _far_grid,
    _weights,
    compute_sensitive_volume,
)
from plots.main import _apply_vetos
from priors.priors import end_o3_ratesandpops

MASS_COMBOS = [(35, 35), (35, 20), (20, 20), (20, 10)]
COMBO = [(35, 20)]


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


def test_far_grid(make_background):
    background = make_background(n=2000, Tb=0.25 * SECONDS_PER_YEAR)
    background = background.sort_by("detection_statistic")

    fars, thresholds = _far_grid(background, max_far=365, num_far_points=50)

    assert 0 < len(fars) <= 50
    assert len(fars) == len(thresholds)
    assert np.all(np.diff(fars) > 0)  # ascending
    assert np.all(np.diff(thresholds) <= 0)  # non-increasing
    assert thresholds[0] == background.detection_statistic[-1]
    assert fars[0] == pytest.approx(SECONDS_PER_YEAR / background.Tb)


def _at_redshifts(ledger, z):
    """Put every draw at source-frame masses (35, 20) and redshifts `z`."""
    z = np.asarray(z, dtype=float)
    ledger.redshift = z
    ledger.mass_1, ledger.mass_2 = 35 * (1 + z), 20 * (1 + z)
    return ledger


def _foreground(make_foreground, z):
    """Draws with statistic 2, injected and recovered 20 s apart from t=10."""
    fg = _at_redshifts(make_foreground(n=len(z)), z)
    fg.injection_time = 10 + 20 * np.arange(len(z), dtype=float)
    fg.detection_time = fg.injection_time.copy()
    fg.shift = np.zeros((len(z), 2))
    fg.detection_statistic = np.full(len(z), 2.0)
    return fg


def _background(make_background):
    """Two statistic-1 events at t=10 and t=30 over Tb=100 s."""
    bg = make_background(n=2, Tb=100.0)
    bg.detection_statistic = np.ones(2)
    bg.detection_time = np.array([10.0, 30.0])
    bg.shift = np.zeros((2, 2))
    return bg


def _efficiency(data, source, dt=None):
    """Mean and error at the single threshold 1, as fractions of V0."""
    result = compute_sensitive_volume(
        data, COMBO, source, dt=dt, max_far=SECONDS_PER_YEAR, num_far_points=1
    )
    v0 = _astrophysical_volume(source)
    return result.sv[0, 0] / v0, result.err[0, 0] / v0


def test_err_counts_every_miss(
    make_background, make_foreground, make_rejected
):
    # three equal draws: one detected, one recovered outside dt, one rejected
    fg = _foreground(make_foreground, [0.2, 0.2])
    fg.detection_time[1] += 10
    rejected = _at_redshifts(make_rejected(n=1), [0.2])
    data = AnalysisData(_background(make_background), fg, rejected)
    mu, err = _efficiency(data, end_o3_ratesandpops()[0], dt=0.1)
    np.testing.assert_allclose([mu, err], [1 / 3, np.sqrt(2 / 27)])


def test_weights_reweight_comoving_to_source_frame(
    make_background, make_foreground, make_rejected
):
    data = AnalysisData(
        _background(make_background),
        _foreground(make_foreground, [0.1, 1.0]),
        make_rejected(n=0),
    )
    weights, _ = _weights(data, COMBO, end_o3_ratesandpops()[0], 0.1)
    # end_o3 draws uniform in comoving volume; the target is uniform in
    # source-frame time, so the weights go as 1 / (1 + z)
    expected = np.array([1 / 1.1, 1 / 2.0])
    # bilby tabulates these redshift priors on 1000-point grids
    np.testing.assert_allclose(
        weights[0], expected / expected.sum(), rtol=5e-5
    )


def test_vetoes_remove_rejected_exposure(
    tmp_path, make_background, make_foreground, make_rejected
):
    segments = tmp_path / "segments.json"
    segments.write_text(
        json.dumps({"segments": {"CAT1": {"H1": [[0, 20]], "L1": []}}})
    )
    # injected at t=10 (vetoed) and t=30, both above the threshold
    fg = _foreground(make_foreground, [0.2, 0.2])
    bg, fg, deadtime = _apply_vetos(
        _background(make_background),
        fg,
        ["CAT1"],
        ["H1", "L1"],
        0,
        40,
        veto_segments=segments,
    )
    rejected = _at_redshifts(make_rejected(n=2), [0.2, 0.2])
    data = AnalysisData(bg, fg, rejected, live_fraction=1 - deadtime)
    mu, err = _efficiency(data, end_o3_ratesandpops()[0])
    # vetoes cover half the time, so the detected survivor plus half of
    # the two rejected draws: reduced-exposure efficiency 1/2
    np.testing.assert_allclose([mu, err], [0.5, np.sqrt(0.125)])
