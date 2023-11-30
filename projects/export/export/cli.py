import jsonargparse
import os
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

    if args.outdir is not None:
        os.makedirs(args.outdir, parents=True, exist_ok=True)

    if args.logfile is not None:
        logdir = os.path.dirname(args.logfile)
        os.makedirs(logdir, parents=True, exist_ok=True)
    verbose = args.pop("verbose")
    configure_logging(args.logfile, verbose)

    args = args.as_dict()
    export(**args)


if __name__ == "__main__":
    main()