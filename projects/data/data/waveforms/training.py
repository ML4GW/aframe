from typing import Callable

from data.waveforms.utils import convert_to_detector_frame
from jsonargparse import ArgumentParser
from ledger.injections import IntrinsicParameterSet, IntrinsicWaveformSet


def training_waveforms(
    num_signals: int,
    sample_rate: int,
    waveform_duration: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    waveform_approximant: str,
    coalescence_time: float,
):
    """
    Generates random training waveforms polarizations from a
    distribution over waveform parameters

    Args:
        num_signals:
            The number of signals to generate
        sample_rate:
            Sample rate of timeseries data, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
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
        waveform_approximant:
            Name of the waveform approximant to use.
        coalescence_time:
            Location of the defining point of the signal within
            the generated waveform

    Returns:
        An IntrinsicParameterSet generated from the sampled parameters
    """
    prior, detector_frame_prior = prior()
    samples = prior.sample(num_signals)
    if not detector_frame_prior:
        samples = convert_to_detector_frame(samples)

    for key in ["ra", "dec"]:
        samples.pop(key)

    params = IntrinsicParameterSet(**samples)
    waveforms = IntrinsicWaveformSet.from_parameters(
        params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
        coalescence_time,
    )
    return waveforms


parser = ArgumentParser()
parser.add_function_arguments(training_waveforms)
parser.add_argument("--output_file", "-o", type=str)


def main(args):
    args = args.training_waveforms.as_dict()
    output_file = args.pop("output_file")
    waveforms = training_waveforms(**args)
    chunks = (
        min(64, args["num_signals"]),
        waveforms.get_waveforms().shape[-1],
    )
    waveforms.write(output_file, chunks=chunks)
