from jsonargparse import ArgumentParser

from data.waveforms.rejection import rejection_sample
from ledger.injections import WaveformSet, waveform_class_factory

parser = ArgumentParser()
parser.add_function_arguments(rejection_sample)
parser.add_argument("--output_file", "-o", type=str)


def main(args):
    args = args.validation_waveforms.as_dict()
    output_file = args.pop("output_file")

    cls = waveform_class_factory(
        args["ifos"],
        WaveformSet,
        "IfoWaveformSet",
    )

    parameters, _ = rejection_sample(**args)
    waveform_set = cls(**parameters)
    waveform_set.write(output_file)
