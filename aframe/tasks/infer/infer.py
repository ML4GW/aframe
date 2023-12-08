import os
import subprocess
from pathlib import Path

import law
import luigi
import psutil
from law.contrib import htcondor
from luigi.util import inherits

from aframe.base import AframeSandboxTask
from aframe.tasks.data.timeslide_waveforms import MergeTimeslideWaveforms

INFER_DIR = Path(__file__).parent.parent.parent / "projects" / "infer"


def get_poetry_env(path):
    output = subprocess.check_output(["poetry", "env", "info", "-p"], cwd=path)
    return output.decode("utf-8").strip()


ENV = get_poetry_env(INFER_DIR)


class InferenceParams(law.Task):
    ip_address = luigi.Parameter()
    data_dir = luigi.Parameter()
    inference_sampling_rate = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    cluster_window_length = luigi.FloatParameter()
    integration_window_length = luigi.FloatParameter()
    fduration = luigi.FloatParameter()


@inherits(InferenceParams)
class Infer(law.SandboxTask):
    image = luigi.Parameter()
    model_repo_dir = luigi.Parameter()
    server_log = luigi.Parameter()
    output_dir = luigi.Parameter()
    triton_image = luigi.Parameter()

    sandbox = "venv::{ENV}"

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local, cluster-internal IP address.
        Currently not a general function.
        """
        nets = psutil.net_if_addrs()
        return nets["enp1s0f0"][0].address

    def requires(self):
        pass

    def run(self):
        from hermes.aeriel.serve import serve

        # initialize a triton server contextmanager
        instance = serve(
            model_repo_dir=self.model_repo_dir,
            image=self.triton_image,
            log_file=self.server_log,
            wait=True,
        )
        # enter the context manager, and launch
        # triton client jobs via conder
        with instance:
            yield Clients.req(self)

            # aggregate


@inherits(InferenceParams)
class Clients(htcondor.HTCondorWorkflow, AframeSandboxTask):
    clients_per_gpu = luigi.IntParameter()

    def create_branch_map(self):
        # segments, shifts
        return

    def workflow_requires(self):
        return MergeTimeslideWaveforms()

    @property
    def tmp_dir(self):
        return os.path.join(self.data_dir, f"tmp-{self.branch}")

    @property
    def foreground_output(self):
        return os.path.join(self.tmp_dir, "foreground.h5")

    @property
    def background_output(self):
        return os.path.join(self.tmp_dir, "background.h5")

    def output(self):
        return [
            law.LocalFileTarget(self.foreground_output),
            law.LocalFileTarget(self.background_output),
        ]

    def run(self):
        from infer import infer

        shifts, background_fname, injection_set_fname = self.branch_data
        infer(
            ip=f"{self.ip}:8001",
            model_name="aframe",
            model_version=-1,
            shifts=shifts,
            background_fname=background_fname,
            injection_set_fname=injection_set_fname,
            batch_size=self.batch_size,
            psd_length=self.psd_length,
            fduration=self.fduration,
            inference_sampling_rate=self.inference_sampling_rate,
            integration_window_length=self.integration_window_length,
            cluster_window_length=self.cluster_window_length,
        )
