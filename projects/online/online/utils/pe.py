import logging
from typing import TYPE_CHECKING, Optional

from astropy import cosmology, units as u
import lal
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from amplfi.utils.result import AmplfiResult
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components

torch.set_num_threads(1)

if TYPE_CHECKING:
    from amplfi.train.architectures.flows import FlowArchitecture
    from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten


def get_redshifts(distances, num_pts=10000):
    """
    Compute redshift using the Planck18 cosmology. Implementation
    taken from https://git.ligo.org/emfollow/em-properties/em-bright/-/blob/main/ligo/em_bright/em_bright.py

    This function accepts distance values in Mpc and computes
    redshifts by interpolating the distance-redshift relation.
    This process is much faster compared to astropy.cosmology
    APIs with lesser than a percent difference.
    """
    func = cosmology.Planck18.luminosity_distance
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    z_min = cosmology.z_at_value(func=func, fval=min_dist * u.Mpc)
    z_max = cosmology.z_at_value(func=func, fval=max_dist * u.Mpc)
    z_steps = np.linspace(
        z_min - (0.1 * z_min), z_max + (0.1 * z_max), num_pts
    )
    lum_dists = cosmology.Planck18.luminosity_distance(z_steps)
    s = interp1d(lum_dists, z_steps)
    redshifts = s(distances)
    return redshifts


def filter_samples(samples, parameter_sampler, inference_params):
    net_mask = torch.ones(samples.shape[0], dtype=bool)
    priors = parameter_sampler.priors
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
    posterior_params = ["chirp_mass", "mass_ratio", "luminosity_distance"]

    posterior = {}
    for param in posterior_params:
        idx = inference_params.index(param)
        posterior[param] = samples.T[idx].flatten()

    # add source frame chirp mass information
    z_vals = get_redshifts(posterior["luminosity_distance"].numpy())
    posterior["chirp_mass_source"] = posterior["chirp_mass"] / (1 + z_vals)

    # TODO remove
    posterior["phi"] = ra
    posterior["ra"] = ra
    posterior["dec"] = dec
    posterior["distance"] = posterior["luminosity_distance"]
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        posterior["chirp_mass"], posterior["mass_ratio"]
    )
    posterior["mass_1"] = mass_1
    posterior["mass_2"] = mass_2
    posterior = pd.DataFrame(posterior)

    result = AmplfiResult(
        label=f"{event_time}",
        posterior=posterior,
        search_parameter_keys=inference_params,
    )
    return result


def warmup_amplfi(
    ifos_to_model: dict[
        tuple[str, ...], tuple["FlowArchitecture", "ChannelWiseScaler"]
    ],
    kernel_length: int,
    psd_length: int,
    sample_rate: int,
    highpass: float,
    samples_per_event: int,
    device: torch.device,
    spectral_density: "SpectralDensity",
    lowpass: Optional[float] = None,
    n_iters: int = 10,
):
    for ifos, (amplfi, _) in ifos_to_model.items():
        strain_size = int(sample_rate * kernel_length)
        strain = torch.randn(1, len(ifos), strain_size, device=device)

        psd_size = int(sample_rate * psd_length)
        psd_strain = torch.randn(1, len(ifos), psd_size, device=device)
        psd = spectral_density(psd_strain.double())

        # construct and bandpass asd
        freqs = torch.fft.rfftfreq(strain.shape[-1], d=1 / sample_rate)
        num_freqs = len(freqs)
        psd = torch.nn.functional.interpolate(
            psd, size=(num_freqs,), mode="linear"
        )

        mask = freqs > highpass
        if lowpass is not None:
            mask *= freqs < lowpass

        freqs = freqs[mask]
        psd = psd[:, :, mask]
        asd = torch.sqrt(psd)

        context = (strain, asd)
        logging.info(f"Warming up {''.join([x[0] for x in ifos])} Amplfi")
        times = []
        with torch.no_grad():
            for i in range(n_iters):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                _ = amplfi.sample(
                    samples_per_event,
                    context,
                )
                end_time.record()

                # wait for the events to be recorded
                torch.cuda.synchronize()

                # convert to seconds
                elapsed_time = start_time.elapsed_time(end_time) / 1000
                times.append(elapsed_time)
                logging.info(f"Iter {i + 1}: {elapsed_time:.2f} s")
        avg_time = np.mean(times[2:])
        logging.info(
            f"Mean time after discarding first two iters: {avg_time:.2f} s"
        )
