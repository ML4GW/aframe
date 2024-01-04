import logging
import os
import subprocess
from pathlib import Path
from typing import TypedDict

import law
import luigi
import psutil
from luigi.util import inherits

from aframe.base import AframeSandboxTask

INFER_DIR = Path(__file__).parent.parent.parent.parent / "projects" / "infer"


def get_poetry_env(path):
    output = None
    try:
        output = subprocess.check_output(
            ["poetry", "env", "info", "-p"], cwd=path
        )
    except subprocess.CalledProcessError:
        logging.warning("Infer directory is not a valid poetry environment")
    else:
        output = output.decode("utf-8").strip()
    return output


class InferRequires(TypedDict):
    data: law.Task
    waveforms: law.Task
    export: law.Task


class InferenceParams(law.Task):
    output_dir = luigi.Parameter()
    ifos = luigi.ListParameter()
    inference_sampling_rate = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    cluster_window_length = luigi.FloatParameter()
    integration_window_length = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    shifts = luigi.ListParameter()
    sequence_id = luigi.IntParameter()
    model_name = luigi.Parameter()
    model_version = luigi.IntParameter()
    triton_image = luigi.Parameter()
    clients_per_gpu = luigi.IntParameter()


@inherits(InferenceParams)
class InferLocal(AframeSandboxTask):
    # dynamically grab poetry environment from
    # local repository directory to use for
    # the sandbox environment of this task
    env_path = get_poetry_env(INFER_DIR)
    sandbox = f"venv::{env_path}"

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local, cluster-internal IP address
        Currently not a general function.
        """
        nets = psutil.net_if_addrs()
        return nets["enp1s0f0"][0].address

    @property
    def num_parallel_jobs(self):
        return self.clients_per_gpu * self.num_gpus

    @property
    def foreground_output(self):
        return os.path.join(self.output_dir, "foreground.hdf5")

    @property
    def background_output(self):
        return os.path.join(self.output_dir, "background.hdf5")

    @property
    def model_repo_dir(self):
        return self.input()["export"].path

    @property
    def background_fnames(self):
        return [
            str(f.path)
            for f in self.input()["data"].collection.targets.values()
        ]

    @property
    def injection_set_fname(self):
        return self.input()["waveforms"][0].path

    def output(self):
        return [
            law.LocalFileTarget(self.foreground_output),
            law.LocalFileTarget(self.background_output),
        ]

    def run(self):
        from infer.deploy import deploy

        from aframe import utils

        segments = utils.segments_from_paths(self.background_fnames)
        num_shifts = utils.get_num_shifts(segments, self.Tb, max(self.shifts))

        deploy(
            ip_address=self.get_ip_address(),
            image=self.triton_image,
            model_name=self.model_name,
            model_repo_dir=self.model_repo_dir,
            ifos=self.ifos,
            shifts=self.shifts,
            num_shifts=num_shifts,
            background_fnames=self.background_fnames,
            injection_set_fname=self.injection_set_fname,
            batch_size=self.batch_size,
            psd_length=self.psd_length,
            fduration=self.fduration,
            inference_sampling_rate=self.inference_sampling_rate,
            integration_window_length=self.integration_window_length,
            cluster_window_length=self.cluster_window_length,
            output_dir=Path(self.output_dir),
            model_version=self.model_version,
            num_parallel_jobs=self.num_parallel_jobs,
        )
