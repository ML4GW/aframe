import os
from pathlib import Path

import law
import luigi
from law.contrib.singularity.config import config_defaults
from luigi.contrib.s3 import S3Client

from aframe.base import AframeSandbox, AframeSingularityTask
from aframe.config import s3
from aframe.tasks.data import DATAFIND_ENV_VARS

root = Path(__file__).resolve().parent.parent.parent


class AframeDataSandbox(AframeSandbox):
    sandbox_type = "aframe_datagen"

    def get_custom_config_section_postfix(self):
        return self.sandbox_type

    @classmethod
    def config(cls):
        config = {}
        default = config_defaults(None).pop("singularity_sandbox")
        default["law_executable"] = "/opt/env/bin/law"
        default["forward_law"] = False
        postfix = cls.sandbox_type
        config[f"singularity_sandbox_{postfix}"] = default
        return config

    @property
    def data_directories(self):
        return ["/cvmfs", "/hdfs", "/gpfs", "/ceph", "/hadoop", "/archive"]

    def _get_volumes(self):
        volumes = super()._get_volumes()

        # bind data directories if they
        # exist on this cluster
        for dir in self.data_directories:
            if os.path.exists(dir):
                volumes[dir] = dir

        # bind aws directory that contains s3 credentials
        aws_dir = os.path.expanduser("~/.aws/")
        volumes[aws_dir] = aws_dir
        return volumes


law.config.update(AframeDataSandbox.config())


class AframeDataTask(AframeSingularityTask):
    job_log = luigi.Parameter(default="")

    @property
    def default_image(self):
        return "data.sif"

    @property
    def client(self):
        return S3Client(endpoint_url=s3().endpoint_url)

    @property
    def sandbox(self):
        return f"aframe_datagen::{self.image}"

    def sandbox_env(self, env):
        env = super().sandbox_env(env)
        # data discovery env vars
        for envvar in DATAFIND_ENV_VARS:
            value = os.getenv(envvar)
            if value is not None:
                env[envvar] = value

        # aws env vars
        env["AWS_ENDPOINT_URL"] = os.getenv("AWS_ENDPOINT_URL")
        return env
