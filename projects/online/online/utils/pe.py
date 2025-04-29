import logging
from typing import TYPE_CHECKING

import lal
import numpy as np
import pandas as pd
import torch
from amplfi.train.data.utils.utils import ParameterSampler
from amplfi.train.priors import precessing_cbc_prior
from amplfi.utils.result import AmplfiResult
from ml4gw.distributions import Cosine
from torch.distributions import Uniform

torch.set_num_threads(1)

if TYPE_CHECKING:
    from amplfi.train.architectures.flows.base import FlowArchitecture
    from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten


def filter_samples(samples, parameter_sampler, inference_params):
    net_mask = torch.ones(samples.shape[0], dtype=bool)
    priors = parameter_sampler.parameters
    for i, param in enumerate(inference_params):
        prior = priors[param]
        curr_samples = samples[:, i]
        log_probs = prior.log_prob(curr_samples)
        mask = log_probs == float("-inf")

        logging.debug(
            f"Removed {mask.sum()}/{len(mask)} samples for parameter "
            f"{param} outside of prior range"
        )

        net_mask &= ~mask

    logging.info(
        f"Removed {(~net_mask).sum()}/{len(net_mask)} total samples "
        f"outside of prior range"
    )
    samples = samples[net_mask]
    return samples


def run_amplfi(
    amplfi_strain,
    amplfi_psd_strain,
    samples_per_event: int,
    spectral_density: "SpectralDensity",
    amplfi_whitener: "Whiten",
    amplfi: "FlowArchitecture",
    std_scaler: "ChannelWiseScaler",
    device: torch.device,
):
    # get pe data from the buffer and whiten it
    amplfi_psd_strain = amplfi_psd_strain.to(device)
    amplfi_strain = amplfi_strain.to(device)[None]
    psd = spectral_density(amplfi_psd_strain)[None]
    whitened = amplfi_whitener(amplfi_strain, psd)

    # construct and bandpass asd
    freqs = torch.fft.rfftfreq(
        whitened.shape[-1], d=1 / amplfi_whitener.sample_rate
    )
    num_freqs = len(freqs)
    psd = torch.nn.functional.interpolate(
        psd, size=(num_freqs,), mode="linear"
    )

    mask = freqs > amplfi_whitener.highpass
    if amplfi_whitener.lowpass is not None:
        mask *= freqs < amplfi_whitener.lowpass

    freqs = freqs[mask]
    psd = psd[:, :, mask]
    asds = torch.sqrt(psd)

    # copy asds for plotting later
    out_asds = asds.clone().detach()

    # sample from the model and descale back to physical units
    logging.info("Starting sampling")
    samples = amplfi.sample(samples_per_event, context=(whitened, asds))
    samples = samples.squeeze(1)
    logging.info("Descaling samples")
    samples = samples.transpose(1, 0)
    descaled_samples = std_scaler(samples, reverse=True)
    descaled_samples = descaled_samples.transpose(1, 0)
    logging.info("Finished AMPLFI")

    return descaled_samples, whitened, out_asds, freqs


def postprocess_samples(
    samples: torch.Tensor,
    event_time: float,
    inference_params: list[str],
    parameter_sampler: torch.nn.Module,
) -> AmplfiResult:
    """
    Process samples into a bilby Result object
    that can be used for all downstream tasks
    """
    samples = filter_samples(samples, parameter_sampler, inference_params)

    phi_idx = inference_params.index("phi")
    dec_idx = inference_params.index("dec")
    ra = torch.remainder(
        lal.GreenwichMeanSiderealTime(event_time) + samples[..., phi_idx],
        torch.as_tensor(2 * torch.pi),
    )
    dec = samples[..., dec_idx]

    # build bilby posterior object for
    # parameters we want to keep
    posterior_params = ["chirp_mass", "mass_ratio", "distance"]

    posterior = {}
    for param in posterior_params:
        idx = inference_params.index(param)
        posterior[param] = samples.T[idx].flatten()

    # TODO remove
    posterior["phi"] = ra
    posterior["ra"] = ra
    posterior["dec"] = dec
    posterior = pd.DataFrame(posterior)

    result = AmplfiResult(
        label=f"{event_time}",
        posterior=posterior,
        search_parameter_keys=inference_params,
    )
    return result


# TODO: need more robust way to
# specify the parameter sampler,
# either from the config,
# or by loading in from checkpoint
def parameter_sampler() -> ParameterSampler:
    base = precessing_cbc_prior()
    base.parameters["chirp_mass"] = Uniform(0, 150, validate_args=False)
    base.parameters["mass_ratio"] = Uniform(0, 0.999, validate_args=False)
    base.parameters["distance"] = Uniform(0, 5000, validate_args=False)
    base.parameters["dec"] = Cosine(validate_args=False)
    base.parameters["phi"] = Uniform(0, 2 * np.pi, validate_args=False)
    base.parameters["psi"] = Uniform(0, np.pi, validate_args=False)
    return base
