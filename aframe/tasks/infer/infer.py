import json
import logging
import os
import socket
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import law
import luigi
import numpy as np
import psutil
from hermes.aeriel.monitor import ServerMonitor
from hermes.aeriel.serve import serve
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.tasks.infer.base import InferBase, InferParameters


@inherits(InferParameters)
class DeployInferLocal(InferBase):
    """
    Launch inference on local gpus
    """

    triton_image = luigi.Parameter()

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local nodes cluster-internal IP address
        """
        for _, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if (
                    addr.family == socket.AF_INET
                    and not addr.address.startswith("127.")
                ):
                    return addr.address
        raise ValueError("No valid IP address found")

    @property
    def model_repo_dir(self):
        return self.input()["model_repository"].path

    def htcondor_workflow_run_context(self):
        """
        Law hook that provides a context manager
        in which the whole workflow is run.

        Return the hermes serve context that will
        spin up a triton and server before the
        actual condor workflow jobs are submitted
        """
        # set the triton server IP address
        # as environment variable with AFRAME prefix
        # so that condor and apptainer will tasks will
        # automatically map it into the environment
        ip = self.get_ip_address()
        os.environ["AFRAME_TRITON_IP"] = ip
        server_log = self.output_dir / "server.log"

        # TODO: figure out why serves
        # `gpus` variable does not expose
        # proper GPU ids to triton
        serve_context = serve(
            self.model_repo_dir,
            self.triton_image,
            log_file=server_log,
            wait=True,
        )

        current_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")

        # helper class to combine
        # the serve and monitor contexts
        class ServerContext:
            def __init__(self, obj):
                self.stack = ExitStack()
                self.obj = obj

            def __enter__(self):
                os.environ["CUDA_VISIBLE_DEVICES"] = self.obj.gpus
                self.stack.enter_context(serve_context)
                monitor = ServerMonitor(
                    model_name=self.obj.model_name,
                    ips="localhost",
                    filename=self.obj.output_dir
                    / f"server-stats-{self.obj.batch_size}.csv",
                    model_version=self.obj.model_version,
                    name="monitor",
                    rate=10,
                )
                time.sleep(1)
                self.stack.enter_context(monitor)

            def __exit__(self, *args):
                self.stack.close()
                os.environ["CUDA_VISIBLE_DEVICES"] = current_gpus

        return ServerContext(self)


@inherits(DeployInferLocal)
class Infer(AframeSingularityTask):
    """
    Law Task that aggregates results from
    individual condor inference jobs
    """

    remove_tmpdir = luigi.BoolParameter(
        description="If `True`, remove directory where individual segment"
        " results are stored after aggregation. Defaults to `True`.",
        default=True,
    )

    @property
    def default_image(self):
        return "infer.sif"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foreground_output = self.output_dir / "foreground.hdf5"
        self.background_output = self.output_dir / "background.hdf5"
        self.zero_lag_output = self.output_dir / "0lag.hdf5"
        self.timeseries_output = self.output_dir / "timeseries.hdf5"

    def output(self):
        output = {}
        output["foreground"] = law.LocalFileTarget(self.foreground_output)
        output["background"] = law.LocalFileTarget(self.background_output)
        if self.zero_lag:
            output["zero_lag"] = law.LocalFileTarget(self.zero_lag_output)
        if self.return_timeseries:
            output["timeseries"] = law.LocalFileTarget(self.timeseries_output)
        return output

    def requires(self):
        # deploy the condor inference jobs;
        # reduce job status poll interval
        # so that jobs can be submitted faster
        return DeployInferLocal.req(
            self,
            request_memory=self.request_memory,
            request_disk=self.request_disk,
            request_cpus=self.request_cpus,
            workflow=self.workflow,
            poll_interval=0.2,
        )

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def background_files(self):
        return np.array(
            [Path(targets["background"].path) for targets in self.targets]
        )

    @property
    def foreground_files(self):
        return np.array(
            [Path(targets["foreground"].path) for targets in self.targets]
        )

    @property
    def metadata_files(self):
        return np.array(
            [Path(targets["metadata"].path) for targets in self.targets]
        )

    @property
    def timeseries_files(self):
        if self.return_timeseries:
            return np.array(
                [Path(targets["timeseries"].path) for targets in self.targets]
            )
        return None

    def get_metadata(self):
        """
        Read in shift and length metadata from the metadata
        files created by each `DeployInferLocal` condor job.
        This data is read from the metadata files rather than
        the hdf5 files because the read operation is O(1000)
        times faster this way
        """
        files = self.metadata_files
        num_files = len(files)
        background_lengths = np.zeros(num_files)
        foreground_lengths = np.zeros(num_files)
        shifts = np.zeros((num_files, len(self.shifts)))
        for i, f in enumerate(files):
            with open(f, "r") as f:
                data = json.load(f)
            background_lengths[i] = data["background_length"]
            foreground_lengths[i] = data["foreground_length"]
            shifts[i] = data["shifts"]

        return background_lengths, foreground_lengths, shifts

    def aggregate_timeseries(self):
        index = []
        with h5py.File(self.timeseries_output, "w") as f:
            ts_group = f.create_group("timeseries")
            for ts in self.timeseries_files:
                with h5py.File(ts, "r") as g:
                    t0 = g.attrs["t0"]
                    shifts = g.attrs["shifts"]
                    background = g["background"][:]
                    foreground = g["foreground"][:]
                    ts_id = f"{t0}_{shifts}"
                    subgroup = ts_group.create_group(ts_id)
                    subgroup.create_dataset("background", data=background)
                    subgroup.create_dataset("foreground", data=foreground)
                    index.append((t0, shifts, f"/timeseries/{ts_id}"))

            # Create an index to make it easier to look up
            # specific segments and shifts
            dtype = np.dtype(
                [
                    ("t0", np.float64),
                    ("shifts", np.int32, (len(shifts),)),
                    ("path", h5py.string_dtype(encoding="utf-8")),
                ]
            )
            index_array = np.array(index, dtype=dtype)
            f.create_dataset("index", data=index_array)

    def run(self):
        import shutil

        from ledger.events import EventSet, RecoveredInjectionSet

        # separate 0lag and background events into different files
        background_lengths, foreground_lengths, shifts = self.get_metadata()
        zero_lag = np.array(
            [all(shift == [0] * len(self.ifos)) for shift in shifts]
        )

        zero_lag_files = self.background_files[zero_lag]
        back_files = self.background_files[~zero_lag]
        zero_lag_length = sum(background_lengths[zero_lag])
        background_length = sum(background_lengths[~zero_lag])
        foreground_length = sum(foreground_lengths)
        foreground_mask = foreground_lengths > 0

        logging.info("Aggregating background files")
        EventSet.aggregate(
            back_files,
            self.background_output,
            clean=False,
            length=background_length,
        )
        logging.info("Aggregating foreground files")
        RecoveredInjectionSet.aggregate(
            self.foreground_files[foreground_mask],
            self.foreground_output,
            clean=False,
            length=foreground_length,
        )
        if len(zero_lag_files) > 0:
            logging.info("Aggregating zero lag files")
            EventSet.aggregate(
                zero_lag_files,
                self.zero_lag_output,
                clean=False,
                length=zero_lag_length,
            )
        if self.return_timeseries:
            logging.info("Aggregating timeseries files")
            self.aggregate_timeseries()

        # Sort background events for later use.
        # TODO: any benefit to sorting foreground for SV calculation?
        if len(back_files) > 0:
            background = EventSet.read(self.background_output)
            background = background.sort_by("detection_statistic")
            background.write(self.background_output)

        if self.remove_tmpdir:
            shutil.rmtree(self.output_dir / "tmp")
