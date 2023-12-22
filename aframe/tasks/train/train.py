import json
import os
import shlex
import sys
from typing import TYPE_CHECKING

import law
import luigi
import yaml
from kr8s.objects import Secret
from luigi.contrib.kubernetes import KubernetesJobTask

from aframe.base import AframeRayTask, AframeSingularityTask, logger
from aframe.config import ray_worker
from aframe.targets import LawS3Target
from aframe.tasks.train.base import RemoteTrainBase, TrainBase
from aframe.tasks.train.config import wandb
from aframe.utils import stream_command

if TYPE_CHECKING:
    from ray_kube import KubernetesRayCluster


class TrainLocal(TrainBase, AframeSingularityTask):
    def sandbox_env(self, _) -> dict[str, str]:
        env = super().sandbox_env(_)
        for key in ["name", "entity", "project", "group", "tags"]:
            value = getattr(wandb(), key)
            if value:
                env[f"WANDB_{key.upper()}"] = value

        return env

    def run(self):
        """
        Run local training in subprocess so that lightning
        can properly handle multi-gpu distribution.
        """

        args = self.get_args()
        if len(self.gpus.split(",")) > 1:
            args.append("--trainer.strategy=ddp")
        cmd = [sys.executable, "-m", "train"] + args
        cmd_str = shlex.join(cmd)
        logger.debug(f"Executing command {cmd_str}")
        stream_command(cmd)

    def output(self):
        dir = law.LocalDirectoryTarget(self.run_dir)
        return dir.child("model.pt", type="f")


class TrainRemote(KubernetesJobTask, RemoteTrainBase):
    image = luigi.Parameter(default="ghcr.io/ml4gw/aframev2/train:main")
    min_gpu_memory = luigi.IntParameter(default=15000)
    request_gpus = luigi.IntParameter(default=4)
    request_cpus = luigi.IntParameter(default=16)
    request_cpu_memory = luigi.Parameter(default="32Gi")

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def kubernetes_namespace(self):
        return "bbhnet"

    @property
    def pod_creation_wait_interal(self):
        return 60

    def get_config(self):
        with open(self.config, "r") as f:
            doc = yaml.safe_load(f)
            json_string = json.dumps(doc)
        return json_string

    def get_args(self):
        # get args from base class, remove the first two
        # which reference the config file, since we'll
        # need to set this as an environment variable of
        # raw yaml content to run remotely.
        args = super().get_args()
        args = args[2:]
        args = ["--config", self.get_config()] + args
        return args

    def output(self):
        return LawS3Target(os.path.join(self.run_dir, "model.pt"))

    @property
    def gpu_constraints(self):
        spec = {}
        spec["affinity"] = {}
        spec["affinity"]["nodeAffinity"] = {}
        spec["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ] = {
            "nodeSelectorTerms": [
                {
                    "matchExpressions": [
                        {
                            "key": "nvidia.com/gpu.memory",
                            "operator": "Gt",
                            "values": [f"{self.min_gpu_memory}"],
                        }
                    ]
                }
            ]
        }
        return spec

    @property
    def backoff_limit(self):
        return 1

    @property
    def secret(self):
        spec = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "s3-credentials", "type": "Opaque"},
        }
        spec["stringData"] = self.get_s3_credentials()

        return Secret(resource=spec)

    @property
    def spec_schema(self):
        spec = self.gpu_constraints
        spec["containers"] = [
            {
                "name": "train",
                "image": self.image,
                "imagePullPolicy": "Always",
                "command": ["python", "-m", "train"],
                "args": self.get_args(),
                "resources": {
                    "limits": {
                        "memory": f"{self.request_cpu_memory}",
                        "cpu": f"{self.request_cpus}",
                        "nvidia.com/gpu": f"{self.request_gpus}",
                    },
                    "requests": {
                        "memory": f"{self.request_cpu_memory}",
                        "cpu": f"{self.request_cpus}",
                        "nvidia.com/gpu": f"{self.request_gpus}",
                    },
                },
                "envFrom": [{"secretRef": {"name": "s3-credentials"}}],
                "env": [
                    {
                        "name": "AWS_ENDPOINT_URL",
                        "value": self.get_internal_s3_url(),
                    },
                    {
                        "name": "WANDB_API_KEY",
                        "value": wandb().api_key,
                    },
                ],
            }
        ]
        return spec

    def run(self):
        self.secret.create()
        super().run()


class TuneRemote(RemoteTrainBase, AframeRayTask):
    search_space = luigi.Parameter()
    num_samples = luigi.IntParameter()
    min_epochs = luigi.IntParameter()
    max_epochs = luigi.IntParameter()
    reduction_factor = luigi.IntParameter()

    def configure_cluster(self, cluster: "KubernetesRayCluster"):
        secret = self.get_s3_credentials()
        cluster.add_secret("s3-credentials", env=secret)
        cluster.set_env("AWS_ENDPOINT_URL", self.get_internal_s3_url())
        return cluster

    def run(self):
        from train.tune import cli as main

        args = self.get_args()
        args.append(f"--tune.num_workers={ray_worker().replicas}")
        args.append(
            "--tune.gpus_per_worker=" + str(ray_worker().gpus_per_worker)
        )
        args.append("--tune.cpus_per_gpu=" + str(ray_worker().cpus_per_gpu))
        args.append("--tune.num_samples={self.num_samples}")
        args.append("--tune.min_epochs={self.min_epochs}")
        args.append("--tune.max_epochs={self.max_epochs}")
        args.append("--tune.reduction_factor={self.reduction_factor}")
        args.append("--tune.storage_dir={self.run_dir}/ray")

        main(args)
