import logging

import bilby
import healpy as hp
import lal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlpe.architectures import MaskedAutoRegressiveFlow, ResNet
from mlpe.injection import nonspin_bbh_chirp_mass_q_parameter_sampler

from ml4gw.transforms import scaler


def get_data_for_pe(
    event_time, input_buffer, fduration, pe_window=4, event_position=3
):
    buffer_start = input_buffer.t0
    sample_rate = input_buffer.sample_rate
    data = input_buffer.input_buffer

    window_start = event_time - buffer_start - event_position - fduration / 2
    window_start = int(sample_rate * window_start)
    window_end = int(window_start + (pe_window + fduration) * sample_rate)

    psd_data = data[:, :window_start]
    pe_data = data[:, window_start:window_end]

    return psd_data, pe_data


def run_amplfi(
    last_event_time,
    input_buffer,
    fduration,
    spectral_density,
    pe_whitener,
    amplfi,
    std_scaler,
    plot_dir,
):
    psd_data, pe_data = get_data_for_pe(
        last_event_time, input_buffer, fduration
    )
    pe_psd = spectral_density(psd_data)
    whitened_pe_data = pe_whitener(pe_data[None], pe_psd[None])
    whitened_pe_data = torch.squeeze(whitened_pe_data)
    if torch.isnan(whitened_pe_data).any():
        logging.info("Whitened data had nan values")
        if torch.isnan(pe_psd).any():
            logging.info("PSD had nan values")

    time = np.arange(last_event_time - 3, last_event_time + 1, 1 / 2048)
    plt.figure()
    plt.plot(time, whitened_pe_data[0].cpu(), label="H1", alpha=0.7)
    plt.plot(time, whitened_pe_data[1].cpu(), label="L1", alpha=0.7)
    plt.legend()
    plt.xlabel("GPS time")
    plt.ylabel("Whitened strain")
    plt.savefig(plot_dir / f"{last_event_time:.2f}.png", dpi=250)
    plt.close()

    res = amplfi.sample(20000, context=whitened_pe_data)
    descaled_samples = std_scaler(res.mT, reverse=True).mT.cpu()
    bilby_res = cast_samples_as_bilby_result(
        descaled_samples[..., :3].numpy(),
        ["chirp_mass", "mass_ratio", "luminosity_distance"],
        f"{last_event_time} result",
    )
    mollview_plot = plot_mollview(
        torch.remainder(
            lal.GreenwichMeanSiderealTime(last_event_time)
            + descaled_samples[..., 7],
            torch.as_tensor(2 * torch.pi),
        )
        - torch.pi,
        descaled_samples[..., 5] + torch.pi / 2,
        title=f"{last_event_time} sky map",
    )
    return bilby_res, mollview_plot


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
    return fig


def set_up_amplfi():
    resnet_context_dim = 20
    resnet_layers = [4, 4, 4]
    resnet_norm_groups = 8
    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
    ]
    num_transforms = 80
    num_blocks = 5
    hidden_features = 120
    embedding = ResNet(
        (2, 8192),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )

    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    flow_obj = MaskedAutoRegressiveFlow(
        (8, 2, 8192),
        embedding,
        None,
        None,
        inference_params,
        prior_func,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features,
    ).to("cuda")

    weights = torch.load(
        "/home/william.benoit/amplfi_models/amplfi-2-det.ckpt"
    )["state_dict"]
    flow_obj.load_state_dict(weights)
    flow_obj.eval()

    std_scaler = scaler.ChannelWiseScaler(8).to("cuda")
    scaler_ckpt = torch.load(
        "/home/william.benoit/amplfi_models/standard-scaler.pth"
    )
    std_scaler.load_state_dict(scaler_ckpt)
    std_scaler.eval()

    return flow_obj, std_scaler
