import shlex
import sys
from collections import defaultdict
from configparser import ConfigParser
from typing import TYPE_CHECKING

import law
import luigi
from luigi.contrib.kubernetes import KubernetesJobTask

from aframe.base import AframeGPUTask, AframeRayTask, logger
from aframe.config import ray_worker
from aframe.tasks.train.base import TrainBase
from aframe.tasks.train.config import nautilus_urls, s3, wandb
from aframe.utils import stream_command

if TYPE_CHECKING:
    from ray_kube import KubernetesRayCluster


class TrainLocal(TrainBase, AframeGPUTask):
    def sandbox_env(self, _) -> dict[str, str]:
        env = super().sandbox_env(_)
        if wandb().api_key:
            env["WANDB_API_KEY"] = wandb().api_key
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
        # TODO: more robust method for finding model.pt
        dir = law.LocalDirectoryTarget(self.run_dir)
        return dir.child("model.pt", type="f")


class TrainRemote(TrainBase, KubernetesJobTask):
    container = luigi.Parameter(default="ghcr.io/ml4gw/aframev2/train:main")
    min_gpu_memory = luigi.IntParameter(default=15000)
    request_gpus = luigi.IntParameter(default=1)
    request_cpus = luigi.IntParameter(default=1)
    request_cpu_memory = luigi.Parameter()

    def get_s3_url(self):
        # if user specified an external nautilus url,
        # map to the corresponding internal url
        url = s3().endpoint_url
        if url in nautilus_urls:
            return nautilus_urls[url]
        return url

    @property
    def gpu_constraints(self):
        spec = defaultdict(dict)
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

    def spec_schema(self):
        spec = self.gpu_constraints
        spec["containers"] = (
            [
                {
                    "name": "train",
                    "image": self.container,
                    "command": ["python", "-m", "train"],
                    "args": self.get_args(),
                    "resources": {
                        "limits": {
                            "memory": self.request_cpu_memory,
                            "cpu": self.request_cpus,
                            "nvidia.com/gpu": self.request_gpus,
                        }
                    },
                }
            ],
        )
        return spec


class TuneRemote(TrainBase, AframeRayTask):
    search_space = luigi.Parameter()
    num_samples = luigi.IntParameter()
    min_epochs = luigi.IntParameter()
    max_epochs = luigi.IntParameter()
    reduction_factor = luigi.IntParameter()

    def configure_cluster(self, cluster: "KubernetesRayCluster"):
        config = ConfigParser.read(s3().credentials)
        keys = ["aws_access_key_id", "aws_secret_access_key"]
        secret = {}
        for key in keys:
            try:
                value = config["default"][key]
            except KeyError:
                raise ValueError(
                    "aws credentials file {} is missing "
                    "key {} in default table".format(s3().credentials, key)
                )
            secret[key] = value
        cluster.add_secret("s3-credentials", env=secret)
        cluster.set_env("AWS_ENDPOINT_URL", self.get_s3_url())
        return cluster

    def get_s3_url(self):
        # if user specified an external nautilus url,
        # map to the corresponding internal url
        url = s3().endpoint_url
        if url in nautilus_urls:
            return nautilus_urls[url]
        return url

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
