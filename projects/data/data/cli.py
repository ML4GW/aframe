from data.authenticate import authenticate
from data.fetch.main import main as fetch
from data.fetch.main import parser as fetch_parser
from data.segments.main import main as query_segments
from data.segments.main import parser as query_parser
from data.timeslide_waveforms import main as generate_timeslide_waveforms
from data.timeslide_waveforms import parser as timeslide_parser
from data.waveforms.main import main as generate_waveforms
from data.waveforms.main import parser as waveform_parser
from jsonargparse import ActionConfigFile, ArgumentParser

from utils.logging import configure_logging


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--log_file", type=str, default=None)
    # parser.add_argument("--verbose", type=bool, default=False)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("query", query_parser)
    subcommands.add_subcommand("fetch", fetch_parser)
    subcommands.add_subcommand("waveforms", waveform_parser)
    subcommands.add_subcommand("timeslide_waveforms", timeslide_parser)

    args = parser.parse_args(args)
    configure_logging(args.log_file)
    authenticate()

    if args.subcommand == "query":
        query_segments(args)

    elif args.subcommand == "fetch":
        fetch(args)

    elif args.subcommand == "waveforms":
        generate_waveforms(args)

    elif args.subcommand == "timeslide_waveforms":
        generate_timeslide_waveforms(args)


if __name__ == "__main__":
    main()
