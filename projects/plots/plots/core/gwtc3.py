import logging
from pathlib import Path

import astropy.cosmology as cosmo
import astropy.units as u
import h5py
import numpy as np
import scipy.stats as stats
from astropy.utils.data import download_file
from utils.cosmology import DEFAULT_COSMOLOGY

catalog_results = {
    "GstLAL": {
        "Tb": 149.3747,
        "vt": {
            (35, 35): 4.1,
            (35, 20): 2.3,
            (20, 20): 1.34,
            (20, 10): 0.6,
            (1.5, 1.5): 2.7e-3,
        },
        "dash": (2, 2),
    },
    "PyCBC-BBH": {
        "Tb": 149.3747,
        "vt": {
            (35, 35): 4.3,
            (35, 20): 2.5,
            (20, 20): 1.42,
            (20, 10): 0.65,
            (1.5, 1.5): 0,
        },
        "dash": (2, 4),
    },
    "cWB": {
        "Tb": 149.3747,
        "vt": {
            (35, 35): 2.6,
            (35, 20): 1.35,
            (20, 20): 0.56,
            (20, 10): 0.24,
            (1.5, 1.5): 0,
        },
        "dash": (4, 2),
    },
    "MBTA": {
        "Tb": 149.3747,
        "vt": {
            (35, 35): 3.3,
            (35, 20): 1.8,
            (20, 20): 1.1,
            (20, 10): 0.51,
            (1.5, 1.5): 3.4e-3,
        },
        "dash": "dashdot",
    },
}

# Most of the following code taken from:
# https://git.ligo.org/tri.nguyen/o3b-catalog-vt/-/blob/master/o3b-vt-dr.ipynb


INJECTION_URL = (
    "https://zenodo.org/records/7890437/files/"
    "endo3_mixture-LIGO-T2100113-v12-1256655642-12905976.hdf5"
)


def get_injection_data(
    pipelines: list[str],
    detection_criterion: str,
    injection_file: Path | None = None,
):
    injection_params = {}

    if injection_file is None:
        logging.info(
            "Downloading injection file from Zenodo, "
            "or reading from cache if exists"
        )
        # will download to ~/.aframe/cache/ if not already downloaded
        injection_file = download_file(
            INJECTION_URL, cache=True, pkgname="aframe"
        )
    else:
        logging.info(f"Reading injection file from {injection_file}")

    with h5py.File(injection_file, "r") as f:
        T_obs = f.attrs["analysis_time_s"] / (365.25 * 24 * 3600)  # years
        N_draw = f.attrs["total_generated"]

        injection_params["m1"] = f["injections/mass1_source"][:]
        injection_params["m2"] = f["injections/mass2_source"][:]
        injection_params["s1x"] = f["injections/spin1x"][:]
        injection_params["s1y"] = f["injections/spin1y"][:]
        injection_params["s1z"] = f["injections/spin1z"][:]
        injection_params["s2x"] = f["injections/spin2x"][:]
        injection_params["s2y"] = f["injections/spin2y"][:]
        injection_params["s2z"] = f["injections/spin2z"][:]
        injection_params["z"] = f["injections/redshift"][:]

        p_draw = f["injections/sampling_pdf"][:]

        det_stat = {}
        for p in pipelines:
            det_stat[p] = f[f"injections/{detection_criterion}_{p}"][:]

    return T_obs, N_draw, injection_params, p_draw, det_stat


def logprob_mass2_lognorm(m2, m1, params):
    """evaluate p(m2 | m1) = c * lognormal(m2 | m, sigma)
    where
    - lognormal is log-normal distribution that is truncated at m1
    - c is the normalization correction factor
    """
    m2_mean = params["m2_mean"]
    sig_lognorm_m1 = params["sig_lognorm_m1"]
    sig_lognorm_m2 = params["sig_lognorm_m2"]

    logc = -stats.lognorm.logcdf(m1, sig_lognorm_m1, scale=m2_mean)
    return np.where(
        m2 < m1,
        stats.lognorm.logpdf(m2, sig_lognorm_m2, scale=m2_mean) + logc,
        -np.inf,
    )


def logprob_mass_lognorm(m1, m2, params):
    """evaluate p(m1, m2) = p(m1) p(m2 | m1)
    with
    - p(m1) = log_normal(m1; m, 0.1)
    - p(m2 | m1) = log_normal(m2; m, 0.1) truncated such that m2 < m1
    """

    sig_lognorm = params["sig_lognorm_m1"]
    m1_mean = params["m1_mean"]

    log_pm1 = stats.lognorm.logpdf(m1, sig_lognorm, scale=m1_mean)
    log_pm2 = logprob_mass2_lognorm(m2, m1, params)

    return log_pm1 + log_pm2


def logprob_spin(sx, sy, sz, params):
    """
    Evaluate p(sx, sy, sz) = (1. / |s|^2) p(|s|, cos theta, phi)
                           = 1. / (4 pi s_max |s|^2)
    where:
    - |s| = sqrt(sx^2 + sy^2 + sz^2)

    The mass `m` determines which s_max to use
    """
    smax = params["smax"]
    s2 = sx**2 + sy**2 + sz**2
    return np.where(
        s2 < smax**2, -np.log(4 * np.pi) - np.log(smax) - np.log(s2), -np.inf
    )


def log_dNdm1dm2ds1ds2dz(
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, z, logprob_mass, logprob_spin, params
):
    """Calculate dN / dm1 dm2 ds1 ds2 dz for selected injections

    Arguments:
    - m1, m2: primary and secondary spin components
    - s1x, s1y, s1z: primary spin components
    - s2x, s2y, s2z: secondary spin components
    - z: redshift
    - logprob_mass: function that takes in m1, m2 and calculate log p(m1, m2)
    - logprob_spin: function that takes in spin params and calculate log p(s)
    - selection: selection function
    - params: parameters for distribution func
    """

    log_pm = logprob_mass(m1, m2, params)  # mass distribution p(m1, m2)

    # primary spin distribution
    s1_max = np.where(m1 < 2, params["smax_ns"], params["smax_bh"])
    spin1_params = params.copy()
    spin1_params["smax"] = s1_max
    log_ps1 = logprob_spin(s1x, s1y, s1z, spin1_params)

    # secondary spin distribution
    s2_max = np.where(m2 < 2, params["smax_ns"], params["smax_bh"])
    spin2_params = params.copy()
    spin2_params["smax"] = s2_max
    log_ps2 = logprob_spin(s2x, s2y, s2z, spin2_params)

    # total spin distribution
    log_ps = log_ps1 + log_ps2

    # Calculate the redshift terms, ignoring rate R0 because it will cancel out
    # dN / dz = dV / dz  * 1 / (1 + z) + (1 + z)^kappa
    # where the second term is for time dilation
    # ignoring the rate because it will cancel out anyway
    cosmo = params["cosmo"]
    log_dNdV = 0
    log_dVdz = np.log(4 * np.pi) + np.log(
        cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value
    )
    log_time_dilation = -np.log(1 + z)
    log_dNdz = log_dNdV + log_dVdz + log_time_dilation

    return log_pm + log_dNdz + log_ps


def logdiffexp(x, y):
    """Evaluate log(exp(x) - exp(y))"""
    return x + np.log1p(-np.exp(y - x))


def _cumulative_logsumexp(x: np.ndarray, reverse: bool) -> np.ndarray:
    """`log(sum(exp(x)))` accumulated forward or in reverse."""
    n = len(x)
    out = np.full(n + 1, -np.inf)
    if n == 0:
        return out
    with np.errstate(divide="ignore"):
        if not reverse:
            out[1:] = np.log(np.cumsum(np.exp(x)))
        else:
            out[:n] = np.log(np.cumsum(np.exp(x)[::-1]))[::-1]
    return out


def _vectorized_pipeline_sv(
    det_stat_p: np.ndarray,
    log_dNs: list[np.ndarray],
    mass_combos: list[tuple],
    p_draw: np.ndarray,
    T_obs: float,
    N_draw: float,
    thresholds: np.ndarray,
    criterion: str,
) -> tuple[dict, dict]:
    """Vectorized SV/err for every mass combo of one pipeline.

    `far` selects a growing forward sum (`det_stat < thresh`, ascending
    sort); `pastro` selects a shrinking reverse sum (`det_stat >
    thresh`), so it needs `side="right"` and `reverse=True`.
    """
    order = np.argsort(det_stat_p)
    ds_sorted = det_stat_p[order]
    n = len(ds_sorted)
    reverse = criterion == "pastro"

    if criterion == "far":
        idxs = np.searchsorted(ds_sorted, thresholds, side="left")
        k = idxs
    else:
        idxs = np.searchsorted(ds_sorted, thresholds, side="right")
        k = n - idxs

    log_T = np.log(T_obs)
    log_N = np.log(N_draw)
    log_p_draw_sorted = np.log(p_draw[order])

    sv, err = {}, {}
    for (m1, m2), log_dN in zip(mass_combos, log_dNs, strict=True):
        key = f"{m1}-{m2}"
        x = log_dN[order] - log_p_draw_sorted
        csum_x = _cumulative_logsumexp(x, reverse=reverse)
        csum_2x = _cumulative_logsumexp(2 * x, reverse=reverse)

        log_VT = log_T - log_N + csum_x[idxs]
        log_s2 = 2 * log_T - 2 * log_N + csum_2x[idxs]

        with np.errstate(invalid="ignore"):
            log_sig2 = logdiffexp(log_s2, 2 * log_VT - log_N)
        log_sig2 = np.where(k == 0, -np.inf, log_sig2)

        sv[key] = np.exp(log_VT) / T_obs
        err[key] = np.exp(log_sig2 / 2) / T_obs

    return sv, err


def get_logdNs(
    mass_combos: list[tuple[float]],
    injection_params: dict,
    sig_lognorm: float,
    smax_ns: float,
    smax_bh: float,
    cosmology: cosmo.Cosmology,
):
    log_dNs = []
    for m1_mean, m2_mean in mass_combos:
        pop_params = {
            "m1_mean": m1_mean,
            "m2_mean": m2_mean,
            "sig_lognorm_m1": sig_lognorm,
            "sig_lognorm_m2": sig_lognorm,
            "smax_ns": smax_ns,
            "smax_bh": smax_bh,
            "cosmo": cosmology,
        }

        log_dNs.append(
            log_dNdm1dm2ds1ds2dz(
                **injection_params,
                logprob_mass=logprob_mass_lognorm,
                logprob_spin=logprob_spin,
                params=pop_params,
            )
        )
    return log_dNs


def _write_result_file(
    output_dir: Path,
    detection_criterion: str,
    detection_thresholds: np.ndarray,
    pipelines: list[str],
    mass_combos: list[tuple],
    sv: dict,
    err: dict,
) -> None:
    """Write the `gwtc-3_pipeline_sv.hdf5` data file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "gwtc-3_pipeline_sv.hdf5"
    with h5py.File(outfile, "w") as f:
        f.create_dataset(f"{detection_criterion}", data=detection_thresholds)
        for p in pipelines:
            g = f.create_group(p)
            for m1, m2 in mass_combos:
                h = g.create_group(f"{m1}-{m2}")
                h.create_dataset("sv", data=np.array(sv[p][f"{m1}-{m2}"]))
                h.create_dataset("err", data=np.array(err[p][f"{m1}-{m2}"]))


def main(
    mass_combos: list[float],
    detection_criterion: str,
    detection_thresholds: list[float],
    output_dir: Path,
    injection_file: Path | None = None,
    pipelines: list[str] = None,
    sig_lognorm: float = 0.1,
    smax_ns: float = 0.4,
    smax_bh: float = 0.998,
    # Note: original used cosmo.FlatwCDM(H0=67.9, Om0=0.3065, w0=-1),
    cosmology: cosmo.Cosmology = DEFAULT_COSMOLOGY,
):
    known_pipelines = ["cwb", "gstlal", "mbta", "pycbc_bbh", "pycbc_hyperbank"]
    if pipelines is None:
        pipelines = known_pipelines
    else:
        unknown_pipelines = set(pipelines) - set(known_pipelines)
        if len(unknown_pipelines) > 0:
            raise ValueError(
                f"Pipelines must be from {known_pipelines}, "
                f"got an unknown pipelines {unknown_pipelines}"
            )

    if detection_criterion not in ["far", "pastro"]:
        raise ValueError(
            "Detection criterion must be either pastro or far, "
            f"got {detection_criterion}"
        )

    (
        T_obs,
        N_draw,
        injection_params,
        p_draw,
        det_stat,
    ) = get_injection_data(pipelines, detection_criterion, injection_file)
    logging.info("Calculating log dNs for all mass combinations")
    log_dNs = get_logdNs(
        mass_combos, injection_params, sig_lognorm, smax_ns, smax_bh, cosmology
    )

    sv, err = {}, {}
    for p in pipelines:
        logging.info(f"Calculating SV for {p}")
        sv[p], err[p] = _vectorized_pipeline_sv(
            det_stat[p],
            log_dNs,
            mass_combos,
            p_draw,
            T_obs,
            N_draw,
            detection_thresholds,
            detection_criterion,
        )

    _write_result_file(
        output_dir,
        detection_criterion,
        detection_thresholds,
        pipelines,
        mass_combos,
        sv,
        err,
    )
    return sv, err
