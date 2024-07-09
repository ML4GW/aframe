import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import jsonargparse
from bokeh.server.server import Server

from .app import App, Data
from .vetos import VetoParser


def normalize_path(path):
    path = Path(path)
    if not path.is_absolute():
        return Path(__file__).resolve().parent / path
    return path


VETO_DEFINER_FILE = normalize_path("./vetos/H1L1-HOFT_C01_O3_CBC.xml")
GATE_PATHS = {
    "H1": normalize_path("./vetos/H1-O3_GATES_1238166018-31197600.txt"),
    "L1": normalize_path("./vetos/L1-O3_GATES_1238166018-31197600.txt"),
}


def main(
    base_dir: Path,
    data_dir: Path,
    ifos: List[str],
    mass_combos: List[tuple],
    source_prior: Callable,
    sample_rate: float,
    fduration: float,
    valid_frac: float,
    port: int = 5005,
    verbose: bool = False,
) -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    data = Data(
        base_dir,
        data_dir,
        mass_combos,
        source_prior,
        ifos,
        sample_rate,
        fduration,
        valid_frac,
    )

    """
    veto_parser = VetoParser(
        VETO_DEFINER_FILE,
        GATE_PATHS,
        data.start,
        data.stop,
        ifos,
    )
    """

    bkapp = App(
        data,
        # veto_parser
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
