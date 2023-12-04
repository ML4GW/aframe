from data.timeslide_waveforms import main as timeslide_waveforms
from jsonargparse import ArgumentParser

parser = ArgumentParser()
parser.add_function_arguments(timeslide_waveforms)


def main(args):
    args = args.timeslide_waveforms.as_dict()
    timeslide_waveforms(args)
