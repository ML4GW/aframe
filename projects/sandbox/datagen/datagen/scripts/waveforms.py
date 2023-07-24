import logging
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from datagen.utils.injection import generate_gw
from typeo import scriptify

from aframe.logging import configure_logging
from aframe.priors.priors import convert_mdc_prior_samples


@scriptify
def main(
    prior: Callable,
    num_signals: int,
    datadir: Path,
    logdir: Path,
    reference_frequency: float,
    minimum_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str = "IMRPhenomPv2",
    force_generation: bool = False,
    verbose: bool = False,
):
    """
    Simulates a set of BBH plus and cross polarization waveforms
    and saves them to an output file.

    Args:
        prior:
            A function that returns a Bilby PriorDict when called
        num_signals:
            Number of waveforms to simulate
        datadir:
            Directory to which the waveforms will be written
        logdir:
            Directory to which the log file will be written
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
            reference to
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        sample_rate:
            Sample rate at which the waveforms will be simulated,
            specified in Hz
        waveform_duration:
            Length of the waveforms in seconds
        waveform_approximant:
            The lalsimulation waveform approximant to use
        force_generation:
            If False, will not generate data if an existing dataset exists
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.

    Returns: The name of the file containing the waveforms and parameters
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

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(num_signals))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
    logging.info("Prior name            : {}".format(prior.__name__))

    # sample gw parameters
    prior, detector_frame_prior = prior()
    params = prior.sample(num_signals)
    params = convert_mdc_prior_samples(params)

    signals = generate_gw(
        params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
        detector_frame_prior,
    )

    # Write params and similar to output file
    if np.isnan(signals).any():
        raise ValueError("The signals contain NaN values")

    with h5py.File(signal_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        for k, v in params.items():
            f.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": num_signals,
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
