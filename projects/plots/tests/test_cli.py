import h5py
import numpy as np
import pytest
from plots import cli
from plots import main as sv_main

MASS_COMBOS = [(35, 35), (35, 20), (20, 20), (20, 10)]
PIPELINES = ["cwb", "gstlal", "mbta", "pycbc_bbh", "pycbc_hyperbank"]


@pytest.fixture(autouse=True)
def stub_gwtc3(monkeypatch):
    """Replace the GWTC-3 comparison curves with flat stand-ins to
    avoid downloading the actual injection set from Zenodo every run.
    """

    def fake(mass_combos, detection_thresholds, **kwargs):
        n = len(detection_thresholds)
        sv = {
            p: {f"{m1}-{m2}": np.full(n, 1.0) for m1, m2 in mass_combos}
            for p in PIPELINES
        }
        err = {
            p: {f"{m1}-{m2}": np.full(n, 0.1) for m1, m2 in mass_combos}
            for p in PIPELINES
        }
        return sv, err

    monkeypatch.setattr(sv_main, "gwtc3_pipeline_sv", fake)


def _run(paths, output_dir):
    """Produce SV outputs from the given analysis files"""
    background, foreground, rejected = paths
    args = [
        "--background",
        str(background),
        "--foreground",
        str(foreground),
        "--rejected_params",
        str(rejected),
        "--ifos",
        "[H1, L1]",
        "--mass_combos",
        str([list(c) for c in MASS_COMBOS]),
        "--source_prior",
        "priors.priors.end_o3_ratesandpops",
        "--output_dir",
        str(output_dir),
    ]
    cli.main(args)


def test_sensitive_volume_end_to_end(analysis_files, tmp_path):
    """Test that `sensitive-volume` produces the expected files/formats."""
    output_dir = tmp_path / "plots"
    _run(analysis_files(), output_dir)

    data = output_dir / "sensitive_volume.hdf5"
    plot = output_dir / "sensitive_volume.html"
    assert data.exists()
    assert plot.exists()
    assert plot.stat().st_size > 0

    with h5py.File(data, "r") as f:
        assert set(f.keys()) == {"thresholds", "fars"} | {
            f"{m1}-{m2}" for m1, m2 in MASS_COMBOS
        }
        n = len(f["fars"][:])
        assert n > 0
        assert f["thresholds"].shape == (n,)

        assert np.all(np.diff(f["fars"][:]) > 0)
        assert np.all(np.diff(f["thresholds"][:]) <= 0)

        for m1, m2 in MASS_COMBOS:
            g = f[f"{m1}-{m2}"]
            assert set(g.keys()) == {"sv", "err"}
            assert g["sv"].shape == (n,)
            assert g["err"].shape == (n,)
            assert np.all(np.isfinite(g["sv"][:]))
            assert np.all(g["err"][:] >= 0)
