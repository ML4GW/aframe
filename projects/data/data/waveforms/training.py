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
