import logging
from typing import TYPE_CHECKING

import bilby
import healpy as hp
import lal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from amplfi.train.priors import precessing_cbc_prior
from amplfi.train.testing import nest2uniq
from astropy import io, table
from astropy import units as u
from ml4gw.distributions import Cosine
from torch.distributions import Uniform

if TYPE_CHECKING:
    from amplfi.train.architectures.flows.base import FlowArchitecture
    from amplfi.train.data.utils.utils import ParameterSampler
    from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten

    from online.utils.buffer import InputBuffer


def filter_samples(samples, parameter_sampler, inference_params):
    net_mask = torch.ones(samples.shape[0], dtype=bool, device=samples.device)
    priors = parameter_sampler.parameters
    for i, param in enumerate(inference_params):
        prior = priors[param]
        samples = samples[:, i]

        mask = (prior.log_prob(samples) == float("-inf")).to(samples.device)
        logging.debug(
            f"Removed {mask.sum()}/{len(mask)} samples for parameter "
            f"{param} outside of prior range"
        )

        net_mask &= ~mask

        logging.info(
            f"Removed {(~net_mask).sum()}/{len(net_mask)} total samples "
            "outside of prior range"
        )
    samples = samples[net_mask]
    return samples


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
    logging.info("Descaling and filtering samples")
    descaled_samples = std_scaler(samples.mT, reverse=True).mT.cpu()

    logging.info("Finished AMPLFI")

    return descaled_samples


def postprocess_samples(
    samples: torch.Tensor,
    event_time: float,
    inference_params: list[str],
    parameter_sampler: torch.nn.Module,
) -> bilby.core.result.Result:
    """
    Process samples into a bilby Result object
    that can be used for all downstream tasks
    """
    # filter samples outside of prior range
    samples = filter_samples(samples, parameter_sampler, inference_params)
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

    posterior = {}
    for param in posterior_params:
        idx = inference_params.index(param)
        posterior[param] = samples.T[idx].flatten()

    posterior["ra"] = ra
    posterior["dec"] = dec
    posterior = pd.DataFrame(posterior)

    result = bilby.result.Result(
        label=f"{event_time}",
        posterior=posterior,
        search_parameter_keys=inference_params,
    )
    return result


def create_histogram_skymap(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
) -> tuple[table.Table, plt.Figure]:
    """Create a skymap from samples of right ascension
    and declination using a naive histogram estimator.
    """
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


# TODO: need more robust way to
# specify the parameter sampler,
# either from the config,
# or by loading in from checkpoint
def parameter_sampler() -> ParameterSampler:
    base = precessing_cbc_prior()
    base.parameters["dec"] = Cosine()
    base.parameters["phi"] = Uniform(0, 2 * np.pi)
    base.parameters["psi"] = Uniform(0, np.pi)
    return base
