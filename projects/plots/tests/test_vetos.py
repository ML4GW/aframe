import numpy as np
import pytest
from plots.vetos import masks as masks_module
from plots.vetos.masks import (
    combine_masks,
    compute_veto_masks,
    load_or_fetch_segments,
)

IFOS = ["H1", "L1"]

SEGMENTS = {
    "CAT1": {"H1": np.array([[1.5, 3.5]]), "L1": np.zeros((0, 2))},
    "CAT2": {"H1": np.zeros((0, 2)), "L1": np.array([[7.5, 8.5]])},
}


def _background(make_background):
    background = make_background(n=10, Tb=10.0, seed=0)
    background.detection_time = np.arange(10, dtype=float)
    background.shift = np.zeros((10, 2))
    return background


def test_compute_veto_masks_known_events(make_background):
    background = _background(make_background)
    result = compute_veto_masks(background, ["CAT1", "CAT2"], IFOS, SEGMENTS)

    expected_cat1 = np.array(
        [False, False, True, True, False, False, False, False, False, False]
    )
    expected_cat2 = np.array([False] * 8 + [True, False], dtype=bool)
    assert (result["CAT1"] == expected_cat1).all()
    assert (result["CAT2"] == expected_cat2).all()


def test_combine(make_background):
    background = _background(make_background)
    result = compute_veto_masks(background, ["CAT1", "CAT2"], IFOS, SEGMENTS)

    assert not combine_masks(result, []).any()
    assert (
        combine_masks(result, ["CAT1", "CAT2"])
        == (result["CAT1"] | result["CAT2"])
    ).all()
    assert (combine_masks(result, ["CAT1"]) == result["CAT1"]).all()

    with pytest.raises(KeyError):
        combine_masks(result, ["UNKNOWN"])


def test_catalog_category(make_background):
    background = _background(make_background)
    segments = {"CATALOG": {ifo: np.array([[4.5, 5.5]]) for ifo in IFOS}}
    result = compute_veto_masks(background, ["CATALOG"], IFOS, segments)
    expected = np.array([False] * 5 + [True] + [False] * 4, dtype=bool)
    assert (result["CATALOG"] == expected).all()


class _FakeVetoParser:
    calls = 0

    def __init__(self, *args, **kwargs):
        _FakeVetoParser.calls += 1

    def get_vetos(self, category):
        return {ifo: np.array([[1.0, 2.0]]) for ifo in IFOS}


def test_load_or_fetch_segments_caches(tmp_path, monkeypatch):
    _FakeVetoParser.calls = 0
    monkeypatch.setattr(masks_module, "VetoParser", _FakeVetoParser)
    cache = tmp_path / "veto_segments.json"

    first = load_or_fetch_segments(["CAT1"], IFOS, 0, 10, cache=cache)
    assert _FakeVetoParser.calls == 1
    assert cache.exists()

    second = load_or_fetch_segments(["CAT1"], IFOS, 0, 10, cache=cache)
    assert _FakeVetoParser.calls == 1, "should reuse the cache, not re-fetch"
    assert (second["CAT1"]["H1"] == first["CAT1"]["H1"]).all()


def test_load_or_fetch_segments_invalidates_on_new_query(
    tmp_path, monkeypatch
):
    _FakeVetoParser.calls = 0
    monkeypatch.setattr(masks_module, "VetoParser", _FakeVetoParser)
    cache = tmp_path / "veto_segments.json"

    load_or_fetch_segments(["CAT1"], IFOS, 0, 10, cache=cache)
    load_or_fetch_segments(["CAT1"], IFOS, 0, 20, cache=cache)
    assert _FakeVetoParser.calls == 2, "different query should re-fetch"
