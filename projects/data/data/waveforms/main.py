import logging
from typing import Callable

from data.waveforms.injection import WaveformGenerator, write_waveforms
from jsonargparse import ArgumentParser

parser = ArgumentParser()
parser.add_function_arguments(WaveformGenerator)
parser.add_argument("--output_file", "-o", type=str)
parser.add_argument("--prior", type=Callable)
parser.add_argument("--num_signals", type=int)


def main(args):
    args = args.waveforms.as_dict()
    output_file = args.pop("output_file")
    prior = args.pop("prior")
    prior, _ = prior()
    num_signals = args.pop("num_signals")

    generator = WaveformGenerator(**args)
    samples = prior.sample(num_signals)
    # TODO: detector frame / source frame logic
    signals = generator(samples)
    logging.info(f"Writing generated waveforms to {output_file}")
    write_waveforms(output_file, signals, samples, generator)
