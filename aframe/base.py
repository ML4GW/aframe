import logging
import os
from collections.abc import Callable
from pathlib import Path

import kr8s
import law
import luigi
from law.contrib import singularity
from law.contrib.singularity.config import config_defaults
from ray_kube import KubernetesRayCluster

from aframe.config import ray_head, ray_worker

root = Path(__file__).resolve().parent.parent
logger = logging.getLogger("luigi-interface")


class AframeSandbox(singularity.SingularitySandbox):
    sandbox_type = "aframe"

    def get_custom_config_section_postfix(self):
        return self.sandbox_type

    @classmethod
    def config(cls):
        config = {}
        default = config_defaults(None).pop("singularity_sandbox")
        default["law_executable"] = "/usr/local/bin/law"
        default["forward_law"] = False
        postfix = cls.sandbox_type
        config[f"singularity_sandbox_{postfix}"] = default
        return config

    def _get_volumes(self):
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[str(root)] = "/opt/aframe"
        return volumes


law.config.update(AframeSandbox.config())


# keep parameters here that
# all tasks should inherit.
# maybe just remove this if
# we're only keeping verbose here
class AframeBase(law.Task):
    verbose = luigi.BoolParameter(default=False)


# base class for tasks that require a container
class AframeSandboxTask(law.SandboxTask, AframeBase):
    dev = luigi.BoolParameter(default=False)
    image = luigi.Parameter()
    container_root = luigi.Parameter(
        default=os.getenv("AFRAME_CONTAINER_ROOT", "")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.isabs(self.image):
            self.image = os.path.join(self.container_root, self.image)

        if not os.path.exists(self.image):
            raise ValueError(
                f"Could not find path to container image {self.image}"
            )

    @property
    def singularity_forward_law(self) -> bool:
        return False

    @property
    def sandbox(self):
        return f"aframe::{self.image}"

    def sandbox_env(self, _):
        env = {}
        for envvar, value in os.environ.items():
            if envvar.startswith("AFRAME_"):
                env[envvar] = value
        # set data and run dirs as env variable in sandbox
        # so they get mapped into the sandbox
        for envvar in ["DATA_DIR", "RUN_DIR"]:
            env[envvar] = os.getenv(envvar, "")


# containerized tasks that require local gpus
class AframeGPUTask(AframeSandboxTask):
    gpus = luigi.Parameter(default="")

    def sandbox_env(self, _):
        env = super().sandbox_env(_)
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
        return env

    @property
    def singularity_args(self) -> Callable:
        def arg_getter():
            if self.gpus:
                return ["--nv"]
            return []

        return arg_getter


# containerized tasks that require a ray cluster
class AframeRayTask(AframeBase):
    container = luigi.Parameter(default="")
    kubeconfig = luigi.Parameter(default="")
    namespace = luigi.Parameter(default="")
    label = luigi.Parameter(default="")

    def configure_cluster(self, cluster):
        return cluster

    def sandbox_before_run(self):
        if not self.container:
            self.cluster = None
            return

        api = kr8s.api(kubeconfig=self.kubeconfig or None)
        num_gpus = ray_worker().gpus
        worker_cpus = ray_worker().cpus_per_gpu * num_gpus
        cluster = KubernetesRayCluster(
            self.container,
            num_workers=ray_worker().replicas,
            worker_cpus=worker_cpus,
            worker_memory=ray_worker().memory,
            gpus_per_worker=num_gpus,
            head_cpus=ray_head().cpus,
            head_memory=ray_head().memory,
            min_gpu_memory=ray_worker().min_gpu_memory,
            api=api,
            label=self.label or None,
        )
        cluster = self.configure_cluster(cluster)

        logger.info("Creating ray cluster")
        cluster.create()
        cluster.wait()
        logger.info("ray cluster online")
        self.cluster = cluster

    def sandbox_after_run(self):
        if self.cluster is not None:
            logger.info("Deleting ray cluster")
            self.cluster.delete()
            self.cluster = None
