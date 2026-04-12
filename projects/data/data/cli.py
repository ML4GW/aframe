from jsonargparse import ActionConfigFile, ArgumentParser

from data.fetch.cli import parser as fetch_parser
from data.fetch.main import main as fetch
from data.segments.cli import parser as query_parser
from data.segments.main import main as query_segments
from data.waveforms.cli import (
    testing_parser,
    training_parser,
    validation_parser,
)
from data.waveforms.testing import main as testing_waveforms
from data.waveforms.training import main as training_waveforms
from data.waveforms.validation import main as validation_waveforms
from utils.logging import configure_logging


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--log_file", type=str, default=None)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("query", query_parser)
    subcommands.add_subcommand("fetch", fetch_parser)
    subcommands.add_subcommand("training_waveforms", training_parser)
    subcommands.add_subcommand("testing_waveforms", testing_parser)
    subcommands.add_subcommand("validation_waveforms", validation_parser)

    args = parser.parse_args(args)
    configure_logging(args.log_file)

    if args.subcommand == "query":
        query_segments(args.query)

    elif args.subcommand == "fetch":
        fetch(args.fetch)

    elif args.subcommand == "training_waveforms":
        training_waveforms(args.training_waveforms)

    elif args.subcommand == "testing_waveforms":
        testing_waveforms(args.testing_waveforms)

    elif args.subcommand == "validation_waveforms":
        validation_waveforms(args.validation_waveforms)


if __name__ == "__main__":
    main()
