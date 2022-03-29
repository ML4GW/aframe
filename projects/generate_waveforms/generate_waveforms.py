#!/usr/bin/env python
# coding: utf-8
import logging
import os

import bilby
import h5py
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from hermes.typeo import typeo

from bbhnet.injection import generate_gw


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

    signals = generate_gw(sample_params, waveform_generator=waveform_generator)

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
