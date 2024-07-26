import os
import socket
from pathlib import Path
from typing import List

import h5py
import law
import luigi
import numpy as np
import psutil
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.tasks.infer.base import InferBase, InferParameters
from hermes.aeriel.serve import serve


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
        # automatically map i
        os.environ["AFRAME_TRITON_IP"] = self.get_ip_address()
        server_log = self.output_dir / "server.log"
        self.ip = self.get_ip_address()
        gpus = [int(gpu) for gpu in self.gpus.split(",")]
        return serve(
            self.model_repo_dir,
            self.triton_image,
            log_file=server_log,
            wait=True,
            gpus=gpus,
        )


@inherits(DeployInferLocal)
class Infer(AframeSingularityTask):
    """
    Law Task that aggregates results from
    individual condor inference jobs
    """

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

        EventSet.aggregate(back_files, self.background_output, clean=True)
        RecoveredInjectionSet.aggregate(
            self.foreground_files, self.foreground_output, clean=True
        )
        if len(zero_lag_files) > 0:
            EventSet.aggregate(
                zero_lag_files, self.zero_lag_output, clean=True
            )

        shutil.rmtree(self.output_dir / "tmp")
