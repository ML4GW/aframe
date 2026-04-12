from jsonargparse import ActionConfigFile, ArgumentParser

from data.waveforms.rejection import rejection_sample
from data.waveforms.testing import testing_waveforms
from data.waveforms.training import training_waveforms
from ledger.injections import WaveformSet, waveform_class_factory

training_parser = ArgumentParser()
training_parser.add_argument("--config", action=ActionConfigFile)
training_parser.add_function_arguments(training_waveforms)
training_parser.add_argument("--output_file", "-o", type=str)

testing_parser = ArgumentParser()
testing_parser.add_argument("--config", action=ActionConfigFile)
testing_parser.add_function_arguments(testing_waveforms)

validation_parser = ArgumentParser()
validation_parser.add_argument("--config", action=ActionConfigFile)
validation_parser.add_function_arguments(rejection_sample)
validation_parser.add_argument("--output_file", "-o", type=str)


def training(args=None):
    cfg = training_parser.parse_args(args)
    args_dict = {k: v for k, v in cfg.as_dict().items() if k != "config"}
    output_file = args_dict.pop("output_file")
    waveforms = training_waveforms(**args_dict)
    chunks = (
        min(64, args_dict["num_signals"]),
        waveforms.get_waveforms().shape[-1],
    )
    waveforms.write(output_file, chunks=chunks)


def testing(args=None):
    cfg = testing_parser.parse_args(args)
    args_dict = {k: v for k, v in cfg.as_dict().items() if k != "config"}
    testing_waveforms(**args_dict)


def validation(args=None):
    cfg = validation_parser.parse_args(args)
    args_dict = {k: v for k, v in cfg.as_dict().items() if k != "config"}
    output_file = args_dict.pop("output_file")
    cls = waveform_class_factory(
        args_dict["ifos"], WaveformSet, "IfoWaveformSet"
    )
    parameters, _ = rejection_sample(**args_dict)
    waveform_set = cls(**parameters)
    waveform_set.write(output_file)
