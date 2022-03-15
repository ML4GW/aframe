#!/usr/bin/env python
# coding: utf-8
import logging
import os

import bilby
import h5py
import numpy as np
import scipy.signal as sig
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from hermes.typeo import typeo


def generate_gw(
    sample_params,
    waveform_generator=None,
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.

    Args:
        sample_params: dictionary of GW parameters
        waveform_generator: bilby.gw.WaveformGenerator with appropriate params

    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples. The first polarization is
        always plus and the second is always cross
    """

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_samples = len(sample_params)

    if waveform_generator is None:
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=8,
            sampling_frequency=16384,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomPv2",
                "reference_frequency": 50,
                "minimum_frequency": 20,
            },
        )

    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration
    waveform_size = int(sample_rate * waveform_duration)

    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    b, a = sig.butter(
        N=8,
        Wn=waveform_generator.waveform_arguments["minimum_frequency"],
        btype="highpass",
        fs=waveform_generator.sampling_frequency,
    )
    for i, p in enumerate(sample_params):
        polarizations = waveform_generator.time_domain_strain(p)
        signals[i] = sig.filtfilt(b, a, list(polarizations.values()), axis=1)

    return signals


@typeo
def main(
    prior_file: str,
    n_samples: int,
    outdir: str,
    waveform_duration: float = 8,
    sample_rate: float = 4096,
):

    """Simulates a set of raw BBH signals and saves them to an output file.

    Args:
        prior_file: prior file for bilby to sample from
        n_samples: number of signal to inject
        outdir: output directory to which signals will be written
        waveform_duration: length of injected waveforms
        sample_rate: sample rate of the signal in Hz

    Returns:
        path to output file
    """

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(n_samples))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
    logging.info("Prior file            : {}".format(prior_file))

    # define a Bilby waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
            "minimum_frequency": 20,
        },
    )

    # sample GW parameters from prior distribution
    priors = bilby.gw.prior.BBHPriorDict(prior_file)
    sample_params = priors.sample(n_samples)

    # generate raw GW signal
    signals = generate_gw(
        sample_params,
        waveform_generator=waveform_generator,
    )

    # Write params and similar to output file
    prior_name = os.path.basename(prior_file)[:-6]
    signal_file = os.path.join(
        outdir, f"signal_file_{prior_name}-{waveform_duration}.h5"
    )

    with h5py.File(signal_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        for k, v in sample_params.items():
            f.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
            }
        )

    return signal_file


if __name__ == "__main__":
    main()
