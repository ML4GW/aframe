from jsonargparse import ActionConfigFile, ArgumentParser

from data.authenticate import authenticate
from data.fetch.main import main as fetch
from data.fetch.main import parser as fetch_parser
from data.segments.main import main as query_segments
from data.segments.main import parser as query_parser
from data.waveforms.testing import main as testing_waveforms
from data.waveforms.testing import parser as testing_parser
from data.waveforms.training import main as training_waveforms
from data.waveforms.training import parser as training_parser
from data.waveforms.validation import main as validation_waveforms
from data.waveforms.validation import parser as validation_parser
from utils.logging import configure_logging


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--log_file", type=str, default=None)
    # parser.add_argument("--verbose", type=bool, default=False)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("query", query_parser)
    subcommands.add_subcommand("fetch", fetch_parser)
    subcommands.add_subcommand("training_waveforms", training_parser)
    subcommands.add_subcommand("testing_waveforms", testing_parser)
    subcommands.add_subcommand("validation_waveforms", validation_parser)

    args = parser.parse_args(args)
    configure_logging(args.log_file)
    authenticate()

    if args.subcommand == "query":
        query_segments(args)

    elif args.subcommand == "fetch":
        fetch(args)

    elif args.subcommand == "training_waveforms":
        training_waveforms(args)

    elif args.subcommand == "testing_waveforms":
        testing_waveforms(args)

    elif args.subcommand == "validation_waveforms":
        validation_waveforms(args)


if __name__ == "__main__":
    main()
