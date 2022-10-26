#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path

import bilby
import h5py
import numpy as np

from bbhnet.injection import generate_gw
from bbhnet.logging import configure_logging
from typeo import scriptify


@scriptify
def main(
    prior_file: Path,
    n_samples: int,
    logdir: Path,
    datadir: Path,
    reference_frequency: float,
    minimum_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str = "IMRPhenomPv2",
    force_generation: bool = False,
    verbose: bool = False,
):
    """Simulates a set of raw BBH signals and saves them to an output file.

    Args:
        prior_file: prior file for bilby to sample from
        n_samples: number of signal to inject
        logdir: directory where log file will be written
        datadir: output directory to which signals will be written
        reference_frequency: reference frequency for waveform generation
        minimum_frequency: minimum_frequency for waveform generation
        sample_rate: rate at which to sample waveforms
        waveform_duration: length of injected waveforms
        waveform_approximant: which lalsimulation waveform approximant to use
        force_generation: if True, generate signals even if path already exists
        verbose: log verbosely
    """

    # make dirs
    datadir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir / "generate_waveforms.log", verbose)

    # check if signal file already exists
    signal_file = datadir / "signals.h5"

    if signal_file.exists() and not force_generation:
        logging.info(
            "Signal data already exists and forced generation is off. "
            "Not generating signals."
        )
        return signal_file

    # if prior file is a relative path,
    # make it relative to this script
    if not prior_file.is_absolute():
        prior_file = Path(__file__).resolve().parent / prior_file

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(n_samples))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
    logging.info("Prior file            : {}".format(prior_file))

    # sample gw parameters from prior distribution
    priors = bilby.gw.prior.ConditionalPriorDict(str(prior_file))
    sample_params = priors.sample(n_samples)

    signals = generate_gw(
        sample_params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
    )

    # Write params and similar to output file
    if np.isnan(signals).any():
        raise ValueError("The signals contain NaN values")

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
                "waveform_approximant": waveform_approximant,
                "reference_frequency:": reference_frequency,
                "minimum_frequency": minimum_frequency,
            }
        )

    return signal_file


if __name__ == "__main__":
    main()
