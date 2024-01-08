import logging
import os
from pathlib import Path
from typing import Optional

import s3fs
from export.main import export

import hermes.quiver as qv


def main(
    weights: str,
    repository_directory: str,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    endpoint_url: str,
    logdir: str,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.TENSORRT,
    clean: bool = False,
    verbose: bool = False,
):
    """
    Export an aframe architecture to a model repository
    and start a triton server to serve the model.

    Used for the remote inference use on a kubernetes cluster,
    where the GPU type is not known ahead of time, and thus
    export can't be done first.
    """

    # pull weights via s3 storing them in
    # a temporary directory where we will also build
    # the model repository
    s3 = s3fs.S3FileSystem(endpoint_url=endpoint_url)
    model_dir = Path("/tmp/weights/")
    model_dir.mkdir(parents=True, exist_ok=True)
    local_weights = model_dir / "weights.pt"
    s3.get_file(str(weights), str(local_weights))

    # first export model to tensorrt, along with
    # constructing snapshotter and whitener
    export(
        local_weights,
        repository_directory,
        Path(logdir),
        num_ifos,
        kernel_length,
        inference_sampling_rate,
        sample_rate,
        batch_size,
        fduration,
        psd_length,
        fftlength,
        highpass,
        streams_per_gpu,
        aframe_instances,
        platform,
        clean,
        verbose,
    )

    logging.info("Starting Triton Server")
    os.environ["LD_LIBRARY_PATH"] = (
        os.environ.get("LD_LIBRARY_PATH", "")
        + ":"
        + os.environ.get("EXTRA_NV_PATHS", "")
    )
    command = [
        "tritonserver",
        "--model-repository",
        "/tmp/model_repo",
        "--repository-poll-secs",
        "30",
        "--model-control-mode",
        "poll",
    ]

    os.execvp(command[0], command)
