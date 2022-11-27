from pathlib import Path
from typing import Dict, List, Optional

from bokeh.server.server import Server
from typeo import scriptify

from bbhnet.logging import configure_logging

from .app import VizApp
from .vetoes import VetoParser


@scriptify
def main(
    ifos: List[str],
    veto_definer_file: Path,
    gate_paths: Dict[str, Path],
    timeslides_results_dir: Path,
    timeslides_strain_dir: Path,
    train_data_dir: Path,
    start: float,
    stop: float,
    sample_rate: float,
    fduration: float,
    valid_frac: float,
    port: int = 5005,
    logdir: Optional[Path] = None,
    verbose: bool = False,
) -> None:

    configure_logging(logdir / "vizapp.log", verbose)

    if not veto_definer_file.is_absolute():
        veto_definer_file = Path(__file__).resolve().parent / veto_definer_file

    for ifo in ifos:
        if not gate_paths[ifo].is_absolute():
            gate_paths[ifo] = Path(__file__).resolve().parent / gate_paths[ifo]

    veto_parser = VetoParser(
        veto_definer_file,
        gate_paths,
        start,
        stop,
        ifos,
    )

    bkapp = VizApp(
        timeslides_results_dir=timeslides_results_dir,
        timeslides_strain_dir=timeslides_strain_dir,
        train_data_dir=train_data_dir,
        veto_parser=veto_parser,
        ifos=ifos,
        sample_rate=sample_rate,
        fduration=fduration,
        valid_frac=valid_frac,
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
