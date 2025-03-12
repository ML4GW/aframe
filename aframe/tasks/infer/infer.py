import os
import socket
import time
from contextlib import ExitStack
from pathlib import Path
from typing import List

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

    def workflow_run_context(self):
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

    def output(self):
        output = {}
        output["foreground"] = law.LocalFileTarget(self.foreground_output)
        output["background"] = law.LocalFileTarget(self.background_output)
        if self.zero_lag:
            output["zero_lag"] = law.LocalFileTarget(self.zero_lag_output)
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

    @classmethod
    def get_shifts(cls, files: List[Path]):
        shifts = []
        for f in files:
            with h5py.File(f) as f:
                shift = f["parameters"]["shift"][0]
                shifts.append(shift)
        return shifts

    def run(self):
        import shutil

        from ledger.events import EventSet, RecoveredInjectionSet

        # separate 0lag and background events into different files
        shifts = self.get_shifts(self.background_files)
        zero_lag = np.array(
            [all(shift == [0] * len(self.ifos)) for shift in shifts]
        )

        zero_lag_files = self.background_files[zero_lag]
        back_files = self.background_files[~zero_lag]

        EventSet.aggregate(back_files, self.background_output, clean=False)
        RecoveredInjectionSet.aggregate(
            self.foreground_files, self.foreground_output, clean=False
        )
        if len(zero_lag_files) > 0:
            EventSet.aggregate(
                zero_lag_files, self.zero_lag_output, clean=False
            )

        # Sort background events for later use.
        # TODO: any benefit to sorting foreground for SV calculation?
        if len(back_files) > 0:
            background = EventSet.read(self.background_output)
            background = background.sort_by("detection_statistic")
            background.write(self.background_output)

        if self.remove_tmpdir:
            shutil.rmtree(self.output_dir / "tmp")
