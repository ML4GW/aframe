import logging
import os
import subprocess
from pathlib import Path

import law
import luigi
import psutil
from kubeml import KubernetesTritonCluster
from luigi.util import inherits

from aframe.base import AframeSandboxTask, S3Task
from aframe.tasks.export.export import ExportParams

INFER_DIR = Path(__file__).parent.parent.parent.parent / "projects" / "infer"


def get_poetry_env(path):
    """
    Get the poetry environment path
    corresponding to a given directory
    """
    output = None
    try:
        output = subprocess.check_output(
            ["poetry", "env", "info", "-p"], cwd=path
        )
    except subprocess.CalledProcessError:
        logging.warning("Infer directory is not a valid poetry environment")
    except FileNotFoundError:
        logging.warning("Infer directory is not a valid poetry environment")
    else:
        output = output.decode("utf-8").strip()
    return output


class InferParameters(law.Task):
    output_dir = luigi.Parameter()
    ifos = luigi.ListParameter(default=["H1", "L1"])
    inference_sampling_rate = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    cluster_window_length = luigi.FloatParameter()
    integration_window_length = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    shifts = luigi.ListParameter(default=[0, 1])
    sequence_id = luigi.IntParameter()
    model_name = luigi.Parameter()
    model_version = luigi.IntParameter()
    clients_per_gpu = luigi.IntParameter()


@inherits(InferParameters)
class InferBase(AframeSandboxTask):
    """
    Base class for inference tasks
    """

    # dynamically grab poetry environment from
    # local repository directory to use for
    # the sandbox environment of this task
    env_path = get_poetry_env(INFER_DIR)
    sandbox = f"venv::{env_path}"

    def get_ip_address(self) -> str:
        raise NotImplementedError

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
    def background_fnames(self):
        return self.input()["data"].collection.targets.values()

    @property
    def injection_set_fname(self):
        return self.input()["waveforms"][0].path

    def output(self):
        return [
            law.LocalFileTarget(self.foreground_output),
            law.LocalFileTarget(self.background_output),
        ]


@inherits(InferParameters)
class InferLocal(InferBase):
    """
    Launch inference on local gpus
    """

    triton_image = luigi.Parameter()

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local, cluster-internal IP address
        Currently not a general function.
        """
        nets = psutil.net_if_addrs()
        return nets["enp1s0f0"][0].address

    @property
    def model_repo_dir(self):
        return self.input()["model_repository"].path

    def run(self):
        from infer.deploy.local import deploy_local

        from aframe import utils

        segments = utils.segments_from_paths(self.background_fnames)
        num_shifts = utils.get_num_shifts(segments, self.Tb, max(self.shifts))
        background_fnames = [f.path for f in self.background_fnames]
        deploy_local(
            ip_address=self.get_ip_address(),
            image=self.triton_image,
            model_name=self.model_name,
            model_repo_dir=self.model_repo_dir,
            ifos=self.ifos,
            shifts=self.shifts,
            num_shifts=num_shifts,
            background_fnames=background_fnames,
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


@inherits(InferParameters, ExportParams)
class InferRemote(InferBase, S3Task):
    """
    Launch inference on a remote kubernetes cluster.
    """

    image = luigi.Parameter()
    replicas = luigi.IntParameter()
    gpus_per_replica = luigi.IntParameter()
    cpus_per_replica = luigi.IntParameter()
    memory = luigi.Parameter(default="8G")
    min_gpu_memory = luigi.Parameter()

    @property
    def command(self):
        return ["python", "-m", "export.remote"]

    @property
    def args(self):
        return [
            "--weights",
            self.weights,
            "--kernel_length",
            str(self.kernel_length),
            "--inference_sampling_rate",
            str(self.inference_sampling_rate),
            "--batch_size",
            str(self.batch_size),
            "--psd_length",
            str(self.psd_length),
            "--streams_per_gpu",
            str(self.streams_per_gpu),
            "--fduration",
            str(self.fduration),
            "--sample_rate",
            str(self.sample_rate),
            "--num_ifos",
            str(len(self.ifos)),
            "--highpass",
            str(self.highpass),
        ]

    def configure_cluster(self, cluster):
        secret = self.get_s3_credentials()
        cluster.add_secret("s3-credentials", env=secret)
        cluster.set_env({"AWS_ENDPOINT_URL": self.get_internal_s3_url()})
        return cluster

    def sandbox_before_run(self):
        cluster = KubernetesTritonCluster(
            self.image,
            self.command,
            self.args,
            self.replicas,
            self.gpus_per_replica,
            self.cpus_per_replica,
            self.memory,
            self.min_gpu_memory,
        )
        cluster.dump("cluster.yaml")
        cluster.create()
        cluster.wait()

    def sandbox_after_run(self):
        """
        Method called after the main `run` method to
        tear down the ray cluster
        """
        if self.cluster is not None:
            self.cluster.delete()
            self.cluster = None

    def get_ip_address(self):
        return self.cluster.get_ip()

    def run(self):
        from infer.deploy.remote import deploy_remote

        from aframe import utils

        segments = utils.segments_from_paths(self.background_fnames)
        num_shifts = utils.get_num_shifts(segments, self.Tb, max(self.shifts))

        background_fnames = [f.path for f in self.background_fnames]
        deploy_remote(
            ip_address=self.get_ip_address(),
            model_name=self.model_name,
            ifos=self.ifos,
            shifts=self.shifts,
            num_shifts=num_shifts,
            background_fnames=background_fnames,
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
