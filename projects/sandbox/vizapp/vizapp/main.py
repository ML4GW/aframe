from pathlib import Path

from bokeh.server.server import Server

from hermes.typeo import typeo

from .app import VizApp


@typeo
def main(
    timeslides_dir: Path,
    data_dir: Path,
    sample_rate: float,
    fduration: float,
    valid_frac: float,
    port: int = 5005,
) -> None:
    bkapp = VizApp(
        timeslides_dir=timeslides_dir,
        data_dir=data_dir,
        sample_rate=sample_rate,
        fduration=fduration,
        valid_frac=valid_frac,
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
