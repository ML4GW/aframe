import logging
import os
import socket
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import jsonargparse
import psutil
from hermes.aeriel.monitor import ServerMonitor
from hermes.aeriel.serve import serve
from utils.logging import configure_logging


def get_ip_address() -> str:
    """
    Get the local node's cluster-internal IP address
    """
    for _, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith(
                "127."
            ):
                return addr.address
    raise ValueError("No valid IP address found")


@contextmanager
def triton_server(
    model_repo_dir: Path,
    triton_image: str,
    gpus: str,
    output_dir: Path,
    model_name: str,
    model_version: int,
    batch_size: int,
):
    """
    Context manager that starts a Triton server and a ServerMonitor.
    Yields the node's cluster-internal IP address.
    """
    server_log = output_dir / "server.log"
    serve_context = serve(
        str(model_repo_dir),
        triton_image,
        log_file=server_log,
        wait=True,
    )

    current_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    try:
        with ExitStack() as stack:
            stack.enter_context(serve_context)
            monitor = ServerMonitor(
                model_name=model_name,
                ips="localhost",
                filename=output_dir / f"server-stats-{batch_size}.csv",
                model_version=model_version,
                name="monitor",
                rate=10,
            )
            time.sleep(1)
            stack.enter_context(monitor)
            yield get_ip_address()
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = current_gpus


def build_parser():
    parser = jsonargparse.ArgumentParser(
        description="Start a Triton inference server and block until done"
    )
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_function_arguments(triton_server)
    parser.add_argument(
        "--ip_file",
        type=Path,
        required=True,
        help="Path where the server IP address will be written.",
    )
    parser.add_argument(
        "--stop_sentinel",
        type=Path,
        required=True,
        help="Path whose existence signals the server to shut down.",
    )
    return parser


def main(args=None):
    """
    Entry point for the start-server CLI.

    Starts a Triton server, writes the node IP to ip_file, then blocks
    until stop_sentinel appears on disk.
    """
    parser = build_parser()
    cfg = parser.parse_args(args)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.logfile is not None:
        os.makedirs(os.path.dirname(cfg.logfile), exist_ok=True)
    configure_logging(cfg.logfile, cfg.verbose)

    logging.info(f"Starting Triton server for model '{cfg.model_name}'")

    server_kwargs = cfg.as_dict()
    for key in ("config", "logfile", "verbose", "ip_file", "stop_sentinel"):
        server_kwargs.pop(key, None)

    with triton_server(**server_kwargs) as ip:
        cfg.ip_file.write_text(ip)
        logging.info(f"Server ready at {ip}; wrote IP to {cfg.ip_file}")
        while not cfg.stop_sentinel.exists():
            time.sleep(5)
        logging.info(
            f"Stop sentinel {cfg.stop_sentinel} detected, shutting down."
        )


if __name__ == "__main__":
    main()
