import os
from pathlib import Path

import law

from aframe.base import AframeSandbox, AframeTask
from aframe.tasks.data import DATAFIND_ENV_VARS

root = Path(__file__).resolve().parent.parent.parent


class AframeDataSandbox(AframeSandbox):
    sandbox_type = "aframe_datagen"

    @property
    def data_directories(self):
        return ["/cvmfs", "/hdfs", "/gpfs", "/ceph", "/hadoop"]

    def _get_volumes(self):
        volumes = super()._get_volumes()

        # bind data directories if they
        # exist on this cluster
        for dir in self.data_directories:
            if os.path.exists(dir):
                volumes[dir] = dir
        return volumes


law.config.update(
    {
        "aframe_datagen": {
            "stagein_dir_name": "stagein",
            "stageout_dir_name": "stageout",
            "law_executable": "/usr/local/bin/law",
        },
        "aframe_datagen_env": {},
        "aframe_datagen_volumes": {},
    }
)


class AframeDataTask(AframeTask):
    @property
    def sandbox(self):
        return f"aframe_datagen::{self.image}"

    def sandbox_env(self, env):
        env = super().sandbox_env(env)
        for envvar in DATAFIND_ENV_VARS:
            value = os.getenv(envvar)
            if value is not None:
                env[envvar] = value
        return env
