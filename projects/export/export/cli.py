import os

import jsonargparse

from export.main import export
from utils.logging import configure_logging


def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_function_arguments(export)
    return parser


def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    logfile = args.pop("logfile")
    args = parser.instantiate_classes(args)
    if logfile is not None:
        logdir = os.path.dirname(logfile)
        os.makedirs(logdir, exist_ok=True)
    verbose = args.pop("verbose")
    configure_logging(logfile, verbose)
    args = args.as_dict()
    # args["platform"] = Platform[args["platform"]]
    export(**args)


if __name__ == "__main__":
    main()
