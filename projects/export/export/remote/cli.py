import os

import jsonargparse
from export.remote.main import export_and_launch_triton

from utils.logging import configure_logging


def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_function_arguments(export_and_launch_triton)
    return parser


def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    logfile = args.pop("logfile")
    if logfile is not None:
        logdir = os.path.dirname(logfile)
        os.makedirs(logdir, exist_ok=True)
    verbose = args.pop("verbose")
    configure_logging(logfile, verbose)
    args = args.as_dict()
    print(args)
    export_and_launch_triton(**args)


if __name__ == "__main__":
    main()
