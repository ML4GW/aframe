from pathlib import Path
from typing import Optional

from bokeh.server.server import Server

from bbhnet.logging import configure_logging
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
    logdir: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    configure_logging(logdir / "vizapp.log", verbose)
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
