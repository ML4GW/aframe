import h5py
import numpy as np
import pytest
from plots.core import gwtc3
from utils.cosmology import DEFAULT_COSMOLOGY

MASS_COMBOS = [(35, 20)]
PIPELINES = ["gstlal"]


def _get_logVT(log_dN, selection, T_obs, N_draw, p_draw):
    log_dN = log_dN[selection]
    p_draw = p_draw[selection]
    if log_dN.size == 0:
        return -np.inf, -np.inf

    log_VT = (
        np.log(T_obs)
        - np.log(N_draw)
        + np.logaddexp.reduce(log_dN - np.log(p_draw))
    )
    log_s2 = (
        2 * np.log(T_obs)
        - 2 * np.log(N_draw)
        + np.logaddexp.reduce(2 * (log_dN - np.log(p_draw)))
    )
    log_sig2 = gwtc3.logdiffexp(log_s2, 2.0 * log_VT - np.log(N_draw))
    return log_VT, log_sig2 / 2


def _compute_sv_loop(
    log_dN, det_stat_p, p_draw, T_obs, N_draw, thresholds, criterion
):
    n = len(thresholds)
    sv = np.zeros(n, dtype=float)
    err = np.zeros(n, dtype=float)
    for i, thresh in enumerate(thresholds):
        if criterion == "far":
            selection = det_stat_p < thresh
        else:
            selection = det_stat_p > thresh
        log_vt, log_sigma_vt = _get_logVT(
            log_dN, selection, T_obs, N_draw, p_draw
        )
        sv[i] = np.exp(log_vt) / T_obs
        err[i] = np.exp(log_sigma_vt) / T_obs
    return sv, err


@pytest.fixture
def injection_file(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    m1 = rng.uniform(20, 60, size=n)
    m2 = rng.uniform(10, m1)
    path = tmp_path / "injections.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["analysis_time_s"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = n * 10
        g = f.create_group("injections")
        g.create_dataset("mass1_source", data=m1)
        g.create_dataset("mass2_source", data=m2)
        for s in ["spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"]:
            g.create_dataset(s, data=rng.uniform(-0.1, 0.1, size=n))
        g.create_dataset("redshift", data=rng.uniform(0.01, 1.0, size=n))
        g.create_dataset("sampling_pdf", data=rng.uniform(0.5, 1.5, size=n))
        for p in PIPELINES:
            g.create_dataset(f"far_{p}", data=rng.exponential(10, size=n))
            g.create_dataset(f"pastro_{p}", data=rng.uniform(0, 1, size=n))
    return path


@pytest.mark.parametrize("criterion", ["far", "pastro"])
def test_gwtc3_vectorized_matches_reference_loop(
    injection_file, tmp_path, criterion
):
    thresholds = np.geomspace(1e-3, 10, 30)

    sv, err = gwtc3.main(
        mass_combos=MASS_COMBOS,
        detection_criterion=criterion,
        detection_thresholds=thresholds,
        output_dir=tmp_path / "run",
        injection_file=injection_file,
        pipelines=PIPELINES,
    )

    T_obs, N_draw, injection_params, p_draw, det_stat = (
        gwtc3.get_injection_data(PIPELINES, criterion, injection_file)
    )
    log_dNs = gwtc3.get_logdNs(
        MASS_COMBOS, injection_params, 0.1, 0.4, 0.998, DEFAULT_COSMOLOGY
    )

    for p in PIPELINES:
        for (m1, m2), log_dN in zip(MASS_COMBOS, log_dNs, strict=True):
            key = f"{m1}-{m2}"
            expected_sv, expected_err = _compute_sv_loop(
                log_dN,
                det_stat[p],
                p_draw,
                T_obs,
                N_draw,
                thresholds,
                criterion,
            )
            np.testing.assert_allclose(sv[p][key], expected_sv, rtol=1e-9)
            np.testing.assert_allclose(err[p][key], expected_err, rtol=1e-9)
