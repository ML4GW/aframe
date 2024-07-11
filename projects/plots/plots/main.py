import logging
import sys
from pathlib import Path
from typing import Callable, List

import jsonargparse
import s3fs
import torch
from architectures.base import Architecture
from bokeh.server.server import Server
from plots.app import App, Data
from plots.vetos import VetoParser

from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


def open_file(path, mode="rb"):
    if str(path).startswith("s3://"):
        fs = s3fs.S3FileSystem()
        return fs.open(path, mode)
    else:
        # For local paths
        return open(path, mode)


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
    weights: str,
    data_dir: Path,
    results_dir: Path,
    ifos: List[str],
    mass_combos: List[tuple],
    source_prior: Callable,
    kernel_length: float,
    psd_length: float,
    highpass: float,
    batch_size: int,
    sample_rate: float,
    inference_sampling_rate: float,
    integration_length: float,
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
    with open_file(weights, "rb") as f:
        weights = torch.load(f, map_location="cpu")["state_dict"]
        weights = {
            k.strip("model."): v
            for k, v in weights.items()
            if k.startswith("model.")
        }
        architecture.load_state_dict(weights)

    architecture = architecture.to(device)
    data = Data(
        data_dir,
        results_dir,
        mass_combos,
        source_prior,
        ifos,
        sample_rate,
        kernel_length,
        psd_length,
        highpass,
        batch_size,
        inference_sampling_rate,
        integration_length,
        fduration,
        valid_frac,
        device,
    )

    whitener = BatchWhitener(
        kernel_length,
        sample_rate,
        inference_sampling_rate,
        batch_size,
        fduration,
        fftlength=2,
        highpass=highpass,
    ).to(device)
    snapshotter = BackgroundSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    ).to(device)

    veto_parser = VetoParser(
        VETO_DEFINER_FILE,
        GATE_PATHS,
        data.start,
        data.stop,
        ifos,
    )

    bkapp = App(data, architecture, whitener, snapshotter, veto_parser)

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


def cli(args=None):
    parser = jsonargparse.ArgumentParser(parser_mode="omegaconf")
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    args.pop("config", None)
    main(**vars(args))


if __name__ == "__main__":
    cli()
