from data.fetch.fetch import fetch
from data.fetch.main import main as _main
from jsonargparse import ActionConfigFile, ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", action=ActionConfigFile)
parser.add_function_arguments(fetch)
parser.add_argument("--output_file", "-o", type=str)


def main(args=None):
    _main(parser.parse_args(args))
