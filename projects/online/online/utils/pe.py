import csv
import logging
import os
import time
from typing import TYPE_CHECKING

import bilby
import healpy as hp
import lal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from amplfi.train.testing import nest2uniq
from astropy import io, table
from astropy import units as u

if TYPE_CHECKING:
    from amplfi.train.architectures.flows.base import FlowArchitecture
    from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten

    from online.utils.buffer import InputBuffer


def run_amplfi(
    event_time: float,
    input_buffer: "InputBuffer",
    samples_per_event: int,
    spectral_density: "SpectralDensity",
    amplfi_whitener: "Whiten",
    amplfi: "FlowArchitecture",
    std_scaler: "ChannelWiseScaler",
    device: torch.device,
):
    # get pe data from the buffer and whiten it
    start = time.time()
    psd_strain, pe_strain = input_buffer.get_amplfi_data(event_time)
    get_strain = time.time()
    psd_strain = psd_strain.to(device)
    pe_strain = pe_strain.to(device)[None]
    to_device = time.time()
    pe_psd = spectral_density(psd_strain)[None]
    get_psd = time.time()
    whitened = amplfi_whitener(pe_strain, pe_psd)
    whiten = time.time()

    # construct and bandpass asd
    freqs = torch.fft.rfftfreq(
        whitened.shape[-1], d=1 / amplfi_whitener.sample_rate
    )
    num_freqs = len(freqs)
    pe_psd = torch.nn.functional.interpolate(
        pe_psd, size=(num_freqs,), mode="linear"
    )

    mask = freqs > amplfi_whitener.highpass
    if amplfi_whitener.lowpass is not None:
        mask *= freqs < amplfi_whitener.lowpass
    pe_psd = pe_psd[:, :, mask]
    asds = torch.sqrt(pe_psd)
    asd_time = time.time()

    # sample from the model and descale back to physical units
    logging.info("Starting sampling")
    samples = amplfi.sample(samples_per_event, context=(whitened, asds))
    sampling = time.time()
    logging.info("Sampling complete")
    descaled_samples = std_scaler(samples.mT, reverse=True).mT.cpu()
    descale = time.time()
    logging.info("Finished AMPLFI")
    fname = "run_amplfi_times.csv"
    with open(fname, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(fname).st_size == 0:
            writer.writerow(
                [
                    "get_strain",
                    "to_device",
                    "get_psd",
                    "whiten",
                    "asd_time",
                    "sampling",
                    "descale",
                ]
            )
        writer.writerow(
            [
                get_strain - start,
                to_device - get_strain,
                get_psd - to_device,
                whiten - get_psd,
                asd_time - whiten,
                sampling - asd_time,
                descale - sampling,
            ]
        )
    return descaled_samples


def skymap_from_samples(
    descaled_samples: torch.Tensor,
    event_time: float,
    inference_params: list[str],
    nside: int,
):
    start = time.time()
    indices = [
        inference_params.index(p)
        for p in ["chirp_mass", "mass_ratio", "distance"]
    ]
    posterior = cast_samples_as_bilby_result(
        descaled_samples[..., indices],
        ["chirp_mass", "mass_ratio", "distance"],
        f"{event_time} result",
    )
    cast_samples = time.time()

    phi_idx = inference_params.index("phi")
    dec_idx = inference_params.index("dec")
    ra = (
        torch.remainder(
            lal.GreenwichMeanSiderealTime(event_time)
            + descaled_samples[..., phi_idx],
            torch.as_tensor(2 * torch.pi),
        )
        - torch.pi
    )
    dec = descaled_samples[..., dec_idx] + torch.pi / 2
    get_ra_dec = time.time()
    skymap, mollview_map = create_skymap(
        ra,
        dec,
        nside,
    )
    create_skymap_time = time.time()
    logging.info("Created skymap")
    fname = "skymap_times.csv"
    with open(fname, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(fname).st_size == 0:
            writer.writerow(
                [
                    "cast_samples",
                    "get_ra_dec",
                    "create_skymap",
                ]
            )
        writer.writerow(
            [
                cast_samples - start,
                get_ra_dec - cast_samples,
                create_skymap_time - get_ra_dec,
            ]
        )

    return posterior, mollview_map, skymap


def cast_samples_as_bilby_result(
    samples,
    inference_params,
    label,
):
    """Cast posterior samples as bilby Result object"""
    posterior = dict()
    for idx, k in enumerate(inference_params):
        posterior[k] = samples.T[idx].flatten()
    posterior = pd.DataFrame(posterior)
    return bilby.result.Result(
        label=label,
        posterior=posterior,
        search_parameter_keys=inference_params,
    )


def create_skymap(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
) -> tuple[table.Table, plt.Figure]:
    """Create a skymap from samples of right ascension and declination."""
    # mask out non physical samples;
    mask = (ra_samples > -np.pi) * (ra_samples < np.pi)
    mask &= (dec_samples > 0) * (dec_samples < np.pi)

    ra_samples = ra_samples[mask]
    dec_samples = dec_samples[mask]
    num_samples = len(ra_samples)

    # calculate number of samples in each pixel
    npix = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)
    uniq_ipix = nest2uniq(nside, np.arange(npix))

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(npix)
    m[np.in1d(range(npix), uniq)] = counts

    post = m / num_samples
    post /= hp.nside2pixarea(nside)
    post /= u.sr

    # convert to astropy table
    t = table.Table(
        [uniq_ipix, post],
        names=["UNIQ", "PROBDENSITY"],
        copy=False,
    )
    fits_table = io.fits.table_to_hdu(t)
    return fits_table, m
