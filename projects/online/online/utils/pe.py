from typing import TYPE_CHECKING

import bilby
import healpy as hp
import lal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from online.utils.buffer import InputBuffer


def run_amplfi(
    event_time: float,
    input_buffer: "InputBuffer",
    samples_per_event: int,
    spectral_density,
    pe_whitener,
    amplfi,
    std_scaler,
    device,
):
    # get pe data from the buffer and whiten it
    psd_strain, pe_strain = input_buffer.get_pe_data(event_time)
    psd_strain = psd_strain.to(device)
    pe_strain = pe_strain.to(device)[None]
    pe_psd = spectral_density(psd_strain)[None]
    whitened = pe_whitener(pe_strain, pe_psd)

    # construct and highpass asd
    freqs = torch.fft.rfftfreq(
        whitened.shape[-1], d=1 / pe_whitener.sample_rate
    )
    num_freqs = len(freqs)
    pe_psd = torch.nn.functional.interpolate(
        pe_psd, size=(num_freqs,), mode="linear"
    )

    mask = freqs > pe_whitener.highpass
    pe_psd = pe_psd[:, :, mask]
    asds = torch.sqrt(pe_psd)

    # sample from the model
    samples = amplfi.sample(samples_per_event, context=(whitened, asds))
    descaled_samples = std_scaler(samples.mT, reverse=True).mT.cpu()
    posterior = cast_samples_as_bilby_result(
        descaled_samples[..., :3].numpy(),
        ["chirp_mass", "mass_ratio", "luminosity_distance"],
        f"{event_time} result",
    )

    phi = (
        torch.remainder(
            lal.GreenwichMeanSiderealTime(event_time)
            + descaled_samples[..., 7],
            torch.as_tensor(2 * torch.pi),
        )
        - torch.pi
    )
    dec = descaled_samples[..., 5] + torch.pi / 2

    skymap = plot_mollview(
        phi,
        dec,
        title=f"{event_time} sky map",
    )
    return posterior, skymap


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


def plot_mollview(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
    fig=None,
    title=None,
):
    # mask out non physical samples;
    ra_samples_mask = (ra_samples > -np.pi) * (ra_samples < np.pi)
    dec_samples_mask = (dec_samples > 0) * (dec_samples < np.pi)

    net_mask = ra_samples_mask * dec_samples_mask
    ra_samples = ra_samples[net_mask]
    dec_samples = dec_samples[net_mask]

    # calculate number of samples in each pixel
    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(NPIX)
    m[np.in1d(range(NPIX), uniq)] = counts

    fig = plt.figure()
    hp.mollview(m, fig=fig, title=title, hold=True)
    plt.close()
    return fig
