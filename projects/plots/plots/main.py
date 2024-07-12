import jsonargparse
from bokeh.server.server import Server
from plots.app import App

from utils.logging import configure_logging


def main(
    app: App,
    verbose: bool = False,
    port: int = 5005,
):
    configure_logging(verbose=verbose)
    server = Server({"/": app}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


def cli(args=None):
    parser = jsonargparse.ArgumentParser(parser_mode="omegaconf")
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    args.pop("config", None)
    main(**vars(args))


if __name__ == "__main__":
    cli()
