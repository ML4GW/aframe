import jsonargparse
from bokeh.server.server import Server
from plots.vizapp.app import App


def main(
    app: App,
    port: int = 5005,
):
    server = Server({"/": app}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


def cli(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_function_arguments(main)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    args.pop("config", None)
    main(**vars(args))


if __name__ == "__main__":
    cli()
