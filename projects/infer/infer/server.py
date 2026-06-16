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
    log_file: Path | None = None,
):
    """
    Context manager that starts a Triton server and a ServerMonitor.
    Yields the node's cluster-internal IP address.
    """
    log_file = (
        Path(log_file) if log_file is not None else output_dir / "server.log"
    )
    serve_context = serve(
        str(model_repo_dir),
        triton_image,
        log_file=log_file,
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


def _activity_since(monitor_csv: Path, pos: int) -> tuple[int, bool]:
    """Read ServerMonitor stats rows appended since byte offset ``pos``.

    Returns the new offset and whether any of those rows recorded inferences.
    """
    if not monitor_csv.exists():
        return pos, False
    with open(monitor_csv, "rb") as f:
        f.seek(pos)
        chunk = f.read()
    for line in chunk.decode(errors="ignore").splitlines():
        cols = line.split(",")
        if len(cols) >= 4:
            try:
                if float(cols[3]) > 0:
                    return pos + len(chunk), True
            except ValueError:
                pass  # header or a partially written row
    return pos + len(chunk), False


def build_parser():
    parser = jsonargparse.ArgumentParser(
        description="Start a Triton inference server and block until done"
    )
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    # log_file is driven by --logfile below, not exposed as its own arg
    parser.add_function_arguments(triton_server, skip={"log_file"})
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
    parser.add_argument(
        "--idle_timeout",
        type=float,
        default=3600.0,
        help="Shut down if no inference is received for this many seconds, so "
        "the server doesn't linger after an interrupted/crashed run. Must "
        "exceed the longest expected gap between requests (e.g. condor "
        "scheduling between branches). Set to 0 to disable.",
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
        os.makedirs(os.path.dirname(cfg.logfile) or ".", exist_ok=True)
    configure_logging(verbose=cfg.verbose)

    logging.info(f"Starting Triton server for model '{cfg.model_name}'")

    server_kwargs = cfg.as_dict()
    for key in (
        "config",
        "logfile",
        "verbose",
        "ip_file",
        "stop_sentinel",
        "idle_timeout",
    ):
        server_kwargs.pop(key, None)

    with triton_server(**server_kwargs, log_file=cfg.logfile) as ip:
        cfg.ip_file.write_text(ip)
        logging.info(f"Server ready at {ip}; wrote IP to {cfg.ip_file}")

        # Use the ServerMonitor's log to detect activity and shut down after
        # if the server has been idle for too long.
        monitor_csv = cfg.output_dir / f"server-stats-{cfg.batch_size}.csv"
        pos = monitor_csv.stat().st_size if monitor_csv.exists() else 0
        last_active = time.monotonic()
        while not cfg.stop_sentinel.exists():
            time.sleep(5)
            pos, active = _activity_since(monitor_csv, pos)
            if active:
                last_active = time.monotonic()
            elif cfg.idle_timeout > 0 and (
                time.monotonic() - last_active > cfg.idle_timeout
            ):
                logging.warning(
                    f"No inference for {cfg.idle_timeout:.0f}s. Shutting down "
                    "to avoid a lingering server."
                )
                return
        logging.info(
            f"Stop sentinel {cfg.stop_sentinel} detected, shutting down."
        )


if __name__ == "__main__":
    main()
