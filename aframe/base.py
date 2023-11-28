import os
from collections.abc import Callable
from pathlib import Path

import law
import luigi
from law.contrib import singularity

from aframe.config import aframe as Config

root = Path(__file__).resolve().parent.parent


class AframeSandbox(singularity.SingularitySandbox):
    sandbox_type = "aframe"

    def _get_volumes(self):
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[str(root)] = "/opt/aframe"
        return volumes


class AframeTask(law.SandboxTask):
    image = luigi.Parameter()
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    cfg = Config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.isabs(self.image):
            self.image = os.path.join(Config().container_root, self.image)

        if not os.path.exists(self.image):
            raise ValueError(
                f"Could not find path to container image {self.image}"
            )

    @property
    def ifos(self):
        return self.cfg.ifos

    @property
    def sandbox(self):
        return f"aframe::{self.image}"

    @property
    def singularity_args(self) -> Callable:
        def arg_getter():
            if self.gpus:
                return ["--nv"]
            return []

        return arg_getter

    def sandbox_env(self, _):
        env = {}
        for envvar, value in os.environ.items():
            if envvar.startswith("AFRAME_"):
                env[envvar] = value

        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
        return env
