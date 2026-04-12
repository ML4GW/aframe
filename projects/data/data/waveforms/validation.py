from data.waveforms.rejection import rejection_sample
from ledger.injections import WaveformSet, waveform_class_factory


def main(args):
    args_dict = {k: v for k, v in args.as_dict().items() if k != "config"}
    output_file = args_dict.pop("output_file")

    cls = waveform_class_factory(
        args_dict["ifos"],
        WaveformSet,
        "IfoWaveformSet",
    )

    parameters, _ = rejection_sample(**args_dict)
    waveform_set = cls(**parameters)
    waveform_set.write(output_file)
