import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from jsonargparse import ArgumentParser

import data.waveforms.utils as utils
from data.waveforms.rejection import rejection_sample
from ledger.injections import InterferometerResponseSet, waveform_class_factory


def testing_waveforms(
    start: float,
    end: float,
    ifos: List[str],
    shifts: List[float],
    spacing: float,
    buffer: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    coalescence_time: float,
    highpass: float,
    snr_threshold: float,
    psd_file: Path,
    output_dir: Path,
    jitter: float = 0.1,
    seed: Optional[int] = None,
):
    """
    Generates testing waveforms via rejection sampling
    for a single segment.

    Args:
        start:
            GPS time of the beginning of the testing segment
        end:
            GPS time of the end of the testing segment
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford. Should be the same length as
            `shifts`
        shifts:
            The length of time in seconds by which each interferometer's
            timeseries will be shifted
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
        prior:
            A function that returns a Bilby PriorDict when called
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
            reference to
        sample_rate:
            Sample rate of timeseries data, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
        waveform_approximant:
            Name of the waveform approximant to use.
        coalescence_time:
            Location of the defining point of the signal within
            the generated waveform
        highpass:
            The frequency to use for a highpass filter, specified
            in Hz
        snr_threshold:
            Minimum SNR of generated waveforms. Sampled parameters
            that result in an SNR below this threshold will be rejected,
            but saved for later use
        psd_file:
            Background file from which to calculate PSDs used for
            estimating waveforms SNR
        output_dir:
            Directory to which the waveform file and rejected parameter
            file will be written
        jitter:
            Scale of random jitter to add to injection times
        seed:
            Random seed to use for reproducibility

    Returns:
        The name of the waveform file and the name of the file containing the
        rejected parameters
    """

    if len(ifos) != len(shifts):
        raise ValueError(
            "Number of ifos must match number of shifts"
            f"got {len(ifos)} ifos and {len(shifts)} shifts"
        )

    # seed process based on start, end and shift
    if seed is not None:
        utils.seed_worker(start, end, shifts, seed)

    # calculate the injection times, determining
    # the number of samples we'll need to generate
    injection_times = utils.calc_segment_injection_times(
        start,
        end - max(shifts),  # TODO: should account for uneven last batch too
        spacing,
        buffer,
        waveform_duration,
    )
    n_samples = len(injection_times)

    # add random jitter to injection times
    jitter = np.random.uniform(-jitter, jitter, size=n_samples)
    injection_times += jitter

    # calculate psd that will be used for snr calculation
    df = 1 / waveform_duration
    logging.info(f"Using background file {psd_file} for psd calculation")
    psds = utils.load_psds(psd_file, ifos, df=df)

    # perform the rejection sampling
    parameters, rejected_params = rejection_sample(
        n_samples,
        prior,
        ifos,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
        coalescence_time,
        highpass,
        snr_threshold,
        psds,
    )

    # create the ResponseSet dataclass based on the passed ifos
    ResponseSet = waveform_class_factory(
        ifos,
        InterferometerResponseSet,
        cls_name="ResponseSet",
    )

    # now, set the injection times and shifts,
    # and create the ResponseSet object
    parameters["injection_time"] = injection_times
    parameters["shift"] = np.array([shifts for _ in range(n_samples)])

    output_dir.mkdir(parents=True, exist_ok=True)
    response_set = ResponseSet(**parameters)
    waveform_fname = output_dir / "waveforms.hdf5"
    utils.io_with_blocking(response_set.write, waveform_fname)

    rejected_fname = output_dir / "rejected-parameters.hdf5"
    utils.io_with_blocking(rejected_params.write, rejected_fname)

    # TODO: compute probability of all parameters against
    # source and all target priors here then save them somehow
    return waveform_fname, rejected_fname


parser = ArgumentParser()
parser.add_function_arguments(testing_waveforms)


def main(args):
    args = args.testing_waveforms.as_dict()
    testing_waveforms(**args)
