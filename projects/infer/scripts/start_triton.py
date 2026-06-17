# ruff: noqa: F821
"""Start the Triton inference server on the submit node.

Launches `start-server` as a detached subprocess so the server keeps running
until stop_triton creates the stop sentinel. Once `start-server` publishes
its IP, it creates the tracking sentinel.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

WAIT_TIMEOUT = 600  # seconds to wait for the server to publish its IP
POLL_INTERVAL = 5  # seconds between checks for the IP file

log = open(snakemake.log[0], "a", buffering=1)
logging.basicConfig(
    stream=log,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
sys.stderr = log

params = snakemake.params
ip_file = Path(params.ip_file)
started = Path(snakemake.output[0])


Path(params.output_dir).mkdir(parents=True, exist_ok=True)

# Clear stale sentinel files before launching a fresh server.
ip_file.unlink(missing_ok=True)
Path(params.stop_sentinel).unlink(missing_ok=True)

# start-server lives in the infer project's environment, so it is launched
# via `uv run --directory projects/infer` rather than imported.
cmd = [
    "uv",
    "run",
    "--directory",
    "projects/infer",
    "start-server",
    "--model_repo_dir",
    snakemake.input.model_repo,
    "--output_dir",
    params.output_dir,
    "--model_name",
    str(params.model_name),
    "--model_version",
    str(params.model_version),
    "--gpus",
    str(params.gpus),
    "--batch_size",
    str(params.batch_size),
    "--triton_image",
    params.triton_image,
    "--ip_file",
    str(ip_file),
    "--stop_sentinel",
    params.stop_sentinel,
    "--logfile",
    params.logfile,
    "--idle_timeout",
    str(params.idle_timeout),
]

# Start a new subprocess detached from this script that will continue running
# after this script exits. The child process inherits a copy of the log file
# descriptor, so it can write to the same log after we close the parent.
subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

deadline = time.monotonic() + WAIT_TIMEOUT
while not ip_file.exists() and time.monotonic() < deadline:
    time.sleep(POLL_INTERVAL)

if not ip_file.exists():
    logging.error(
        f"Triton server not ready within {WAIT_TIMEOUT}s. "
        f"Check the server log: {params.logfile}"
    )
    sys.exit(1)

logging.info(f"Triton server ready at {ip_file.read_text().strip()}")
started.touch()
