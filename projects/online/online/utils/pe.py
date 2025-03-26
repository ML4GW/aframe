import logging
from typing import TYPE_CHECKING

import lal
import pandas as pd
import torch
from amplfi.train.testing import Result

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
    psd_strain, pe_strain = input_buffer.get_amplfi_data(event_time)
    psd_strain = psd_strain.to(device)
    pe_strain = pe_strain.to(device)[None]
    pe_psd = spectral_density(psd_strain)[None]
    whitened = amplfi_whitener(pe_strain, pe_psd)

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

    # sample from the model and descale back to physical units
    logging.info("Starting sampling")
    samples = amplfi.sample(samples_per_event, context=(whitened, asds))
    logging.info("Sampling complete")
    descaled_samples = std_scaler(samples.mT, reverse=True).mT.cpu()
    logging.info("Finished AMPLFI")

    return descaled_samples


def postprocess_samples(
    samples: torch.Tensor, event_time: float, inference_params: list[str]
) -> Result:
    """
    Process samples into a bilby Result object
    that can be used for all downstream tasks
    """
    # convert samples from relative angle phi
    # to physical right ascension value;
    # convert declination from [-pi / 2, pi / 2] -> [0, pi]
    phi_idx = inference_params.index("phi")
    dec_idx = inference_params.index("dec")
    ra = (
        torch.remainder(
            lal.GreenwichMeanSiderealTime(event_time) + samples[..., phi_idx],
            torch.as_tensor(2 * torch.pi),
        )
        - torch.pi
    )
    dec = samples[..., dec_idx] + torch.pi / 2

    # build bilby posterior object for
    # parameters we want to keep
    posterior_params = ["chirp_mass", "mass_ratio", "distance"]

    posterior = dict()
    for param in posterior_params:
        idx = inference_params.index(param)
        posterior[param] = samples.T[idx].flatten()

    posterior["ra"] = ra
    posterior["dec"] = dec
    posterior = pd.DataFrame(posterior)

    result = Result(
        label=f"{event_time}",
        posterior=posterior,
        search_parameter_keys=inference_params,
    )
    return result
