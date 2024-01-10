import logging
import os
from collections.abc import Callable
from pathlib import Path

import kr8s
import law
import luigi
from kubeml import KubernetesRayCluster
from law.contrib import singularity
from law.contrib.singularity.config import config_defaults

from aframe.config import ray_head, ray_worker

root = Path(__file__).resolve().parent.parent
logger = logging.getLogger("luigi-interface")


class AframeSandbox(singularity.SingularitySandbox):
    """
    Base sandbox for running aframe tasks in a singularity container.
    """

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
        # if running in dev mode, mount the local
        # aframe repo into the container so code changes
        # are reflected
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[str(root)] = "/opt/aframe"

        return volumes


# update the law config to let it know about
# the aframe sandbox config changes
law.config.update(AframeSandbox.config())


class AframeSandboxTask(law.SandboxTask):
    """
    Base task for __any__ Sandbox task (e.g. singularity, poetry/venv etc.)

    The `sandbox` property should return a string
    of the form `{sandbox_type}::{path}`.`sandbox_type`
    corresponds to the `sandbox_type` class variable
    of the desired Sandbox class.

    `path` is the path to the specific sandbox image
    (for singularity sandboxes)or environment (when using poetry / venv)
    one wishes to run the task in.
    """

    dev = luigi.BoolParameter(default=False, significant=False)
    gpus = luigi.Parameter(default="", significant=False)

    @property
    def sandbox(self):
        return None

    def sandbox_env(self, _):
        """
        Set local environment variables to be set inside the sandbox
        """

        # set any environment variables
        # that start with AFRAME_
        env = {}
        for envvar, value in os.environ.items():
            if envvar.startswith("AFRAME_"):
                env[envvar] = value

        # default tmpdir in containers was /tmp/,
        # which does not contain that much memory.
        # map in local tmpdir (should be /local/albert.einstein)
        # which has is enough memory to write large temp
        # files with luigi/law
        for envvar in ["TMPDIR"]:
            env[envvar] = os.getenv(envvar, "")

        # if gpus are specified, expose them inside container
        # via CUDA_VISIBLE_DEVICES env variable
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
        return env

    @property
    def num_gpus(self):
        if self.gpus:
            return len(self.gpus.split(","))
        return 0


class AframeSingularityTask(AframeSandboxTask):
    """
    Sandbox task for running aframe tasks in
    a singularity container. Tasks that wish
    to run in a singularity container should
    inherit from this class.
    """

    image = luigi.Parameter(default="")
    container_root = luigi.Parameter(
        default=os.getenv("AFRAME_CONTAINER_ROOT", ""), significant=False
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
    def singularity_args(self) -> Callable:
        def arg_getter():
            if self.gpus:
                return ["--nv"]
            return []

        return arg_getter

    @property
    def sandbox(self):
        # sandbox is singularity sandbox defined above;
        # image is the path to
        # the singularity container .sif file
        return f"aframe::{self.image}"

    def singularity_forward_law(self) -> bool:
        return True


class RayCluster(KubernetesRayCluster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head["spec"]["template"]["spec"]["containers"][0][
            "imagePullPolicy"
        ] = "Always"
        self.worker["spec"]["template"]["spec"]["containers"][0][
            "imagePullPolicy"
        ] = "Always"


class AframeRayTask(AframeSingularityTask):
    """
    Base task for tasks that require a ray cluster
    deployed via kubernetes.

    Tasks that wish to utillize a ray cluster
    (for now, just hyperparameter tuning)
    should inherit from this class.
    """

    container = luigi.Parameter(
        default="ghcr.io/ml4gw/aframev2/train:dev", significant=False
    )
    kubeconfig = luigi.Parameter(default="", significant=False)
    namespace = luigi.Parameter(default="", significant=False)
    label = luigi.Parameter(default="", significant=False)

    def configure_cluster(self, cluster):
        return cluster

    def sandbox_env(self, _):
        # hacky way to pass cluster ip to sandbox task
        # that gets run in the container.
        env = super().sandbox_env(_)
        env["AFRAME_RAY_CLUSTER_IP"] = self.ip
        return env

    def sandbox_before_run(self):
        """
        Method called before the main `run` method to set up
        and launch the ray cluster
        """
        if not self.container:
            self.cluster = None
            return

        api = kr8s.api(kubeconfig=self.kubeconfig or None)

        worker_cpus = ray_worker().cpus_per_gpu * ray_worker().gpus_per_replica
        cluster = KubernetesRayCluster(
            self.container,
            num_workers=ray_worker().replicas,
            worker_cpus=worker_cpus,
            worker_memory=ray_worker().memory,
            gpus_per_worker=ray_worker().gpus_per_replica,
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
        self.ip = cluster.get_ip()

    def sandbox_after_run(self):
        """
        Method called after the main `run` method to
        tear down the ray cluster
        """
        if self.cluster is not None:
            logger.info("Deleting ray cluster")
            self.cluster.delete()
            self.cluster = None
