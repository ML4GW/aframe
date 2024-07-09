import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import jsonargparse
from bokeh.server.server import Server

from .app import App, Data
from .vetos import VetoParser
from utils.preprocessing import BatchWhitener, BackgroundSnapshotter
from architectures.base import Architecture
import torch

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
    architecture: Architecture,
    base_dir: Path,
    data_dir: Path,
    ifos: List[str],
    mass_combos: List[tuple],
    source_prior: Callable,
    kernel_length: int,
    psd_length: int,
    highpass: float,
    batch_size: int,
    sample_rate: float,
    inference_sampling_rate: float,
    fduration: float,
    valid_frac: float,
    port: int = 5005,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )


    # load in best model
    weights = base_dir / "training" / "model.pt"
    checkpoint = torch.load(weights, map_location=device)
    architecture.load_state_dict(checkpoint)

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


    # build modules for performing on the fly inference
    length = kernel_length + fduration + psd_length
    whitener = BatchWhitener(
        length,
        sample_rate,
        inference_sampling_rate,
        batch_size,
        fduration,
        highpass=highpass,
    )
    snapshotter = BackgroundSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
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


def cli(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    cli()



if __name__ == "__main__":
    main()
