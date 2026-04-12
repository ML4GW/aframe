import logging
import os

from data.fetch.fetch import fetch
from jsonargparse import ActionConfigFile, ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", action=ActionConfigFile)
parser.add_function_arguments(fetch)
parser.add_argument("--output_directory", "-o", type=str)
parser.add_argument("--prefix", "-p", type=str, default="background")


def main(args=None):
    cfg = parser.parse_args(args)
    args_dict = {k: v for k, v in cfg.as_dict().items() if k != "config"}

    output_directory = args_dict.pop("output_directory")
    prefix = args_dict.pop("prefix")
    X = fetch(**args_dict)

    duration = args_dict["end"] - args_dict["start"]
    fname = "{}-{}-{}.hdf5".format(
        prefix, int(args_dict["start"]), int(duration)
    )
    fname = os.path.join(output_directory, fname)

    logging.info(f"Writing downloaded data to {fname}")
    X.write(fname, format="hdf5")
