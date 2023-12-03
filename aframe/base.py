import logging
import os
from collections.abc import Callable
from pathlib import Path

import kr8s
import law
import luigi
from law.contrib import singularity
from ray_kube import KubernetesRayCluster

from aframe.config import aframe as Config

root = Path(__file__).resolve().parent.parent
logger = logging.getLogger("luigi-interface")


class AframeSandbox(singularity.SingularitySandbox):
    sandbox_type = "aframe"

    def _get_volumes(self):
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[str(root)] = "/opt/aframe"
        return volumes


law.config.update(
    {
        "aframe": {
            "stagein_dir_name": "stagein",
            "stageout_dir_name": "stageout",
            "law_executable": "/usr/local/bin/law",
        },
        "aframe_env": {},
        "aframe_volumes": {},
    }
)


class AframeTask(law.SandboxTask):
    image = luigi.Parameter()
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    cfg = Config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.isabs(self.image):
            self.image = os.path.join(self.cfg.container_root, self.image)

        if not os.path.exists(self.image):
            raise ValueError(
                f"Could not find path to container image {self.image}"
            )

    @property
    def singularity_forward_law(self) -> bool:
        return False

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
        # set data and run dirs as env variable in sandbox
        # so they get mapped into the sandbox
        for envvar in ["DATA_DIR", "RUN_DIR"]:
            env[envvar] = os.getenv(envvar, "")

        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
        return env


class AframeRayTask(AframeTask):
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
        num_gpus = self.cfg.ray_worker.gpus
        worker_cpus = self.cfg.ray_worker.cpus_per_gpu * num_gpus
        cluster = KubernetesRayCluster(
            self.container,
            num_workers=self.cfg.ray_worker.replicas,
            worker_cpus=worker_cpus,
            worker_memory=self.cfg.ray_worker.memory,
            gpus_per_worker=num_gpus,
            head_cpus=self.cfg.ray_head.cpus,
            head_memory=self.cfg.ray_head.memory,
            min_gpu_memory=self.cfg.ray_worker.min_gpu_memory,
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
