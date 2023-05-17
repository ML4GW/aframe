from pathlib import Path
from typing import Callable, Dict, List, Optional

from bokeh.server.server import Server
from typeo import scriptify

from bbhnet.logging import configure_logging

from .app import VizApp
from .vetoes import VetoParser


def _normalize_path(path: Path):
    if not path.is_absolute():
        return Path(__file__).resolve().parent / path
    return path


@scriptify
def main(
    outdir: Path,
    datadir: Path,
    veto_definer_file: Path,
    gate_paths: Dict[str, Path],
    ifos: List[str],
    cosmology: Callable,
    source_prior: Callable,
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

    veto_definer_file = _normalize_path(veto_definer_file)
    for ifo in ifos:
        gate_paths[ifo] = _normalize_path(gate_paths[ifo])

    veto_parser = VetoParser(
        veto_definer_file,
        gate_paths,
        start,
        stop,
        ifos,
    )

    cosmology = cosmology()

    bkapp = VizApp(
        base_directory=outdir,
        data_directory=datadir,
        cosmology=cosmology,
        source_prior=source_prior,
        ifos=ifos,
        sample_rate=sample_rate,
        fduration=fduration,
        valid_frac=valid_frac,
        veto_parser=veto_parser,
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
