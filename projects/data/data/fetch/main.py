import logging
import os

from data.fetch.fetch import fetch
from jsonargparse import ArgumentParser

parser = ArgumentParser()
parser.add_function_arguments(fetch)
parser.add_argument("--output_directory", "-o", type=str)
parser.add_argument("--prefix", "-p", type=str, default="background")


def main(args=None):
    args = args.fetch.as_dict()
    output_directory = args.pop("output_directory")
    prefix = args.pop("prefix")
    X = fetch(**args)

    duration = args["end"] - args["start"]
    fname = "{}-{}-{}.hdf5".format(prefix, int(args["start"]), int(duration))
    fname = os.path.join(output_directory, fname)

    logging.info(f"Writing downloaded data to {fname}")
    X.write(fname, format="hdf5")
