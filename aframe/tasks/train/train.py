import json
import os
import shlex
import sys
from typing import TYPE_CHECKING, Dict

import law
import luigi
import yaml
from kr8s.objects import Secret
from luigi.contrib.kubernetes import KubernetesJobTask

from aframe.base import AframeRayTask, AframeSingularityTask, logger
from aframe.config import ray_worker, s3
from aframe.targets import LawS3Target
from aframe.tasks.train.base import RemoteTrainBase, TrainBase
from aframe.tasks.train.config import train_remote, wandb
from aframe.utils import stream_command

if TYPE_CHECKING:
    from ray_kube import KubernetesRayCluster


class TrainLocal(TrainBase, AframeSingularityTask):
    @property
    def default_image(self):
        return "train.sif"

    def sandbox_env(self, _) -> Dict[str, str]:
        env = super().sandbox_env(_)
        for key in ["name", "entity", "project", "group", "tags", "api_key"]:
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
    image = luigi.Parameter(default=train_remote().image)
    min_gpu_memory = luigi.IntParameter(default=train_remote().min_gpu_memory)
    request_gpus = luigi.IntParameter(default=train_remote().request_gpus)
    request_cpus = luigi.IntParameter(default=train_remote().request_cpus)
    request_cpu_memory = luigi.Parameter(
        default=train_remote().request_cpu_memory
    )

    @property
    def default_image(self):
        return None

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def kubernetes_namespace(self):
        return "bbhnet"

    @property
    def pod_creation_wait_interal(self):
        return 120

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
        spec["stringData"] = s3().get_s3_credentials()

        return Secret(resource=spec)

    @property
    def spec_schema(self):
        spec = self.gpu_constraints
        spec["containers"] = [
            {
                "name": "train",
                "image": self.image,
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
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
                        "value": s3().get_internal_s3_url(),
                    },
                    {
                        "name": "WANDB_API_KEY",
                        "value": wandb().api_key,
                    },
                ],
            }
        ]

        spec["volumes"] = [
            {
                "name": "dshm",
                "emptyDir": {"sizeLimit": "32Gi", "medium": "Memory"},
            }
        ]
        return spec

    def run(self):
        self.secret.create()
        super().run()

    def on_failure(self, exc):
        self.secret.delete()
        super().on_failure(exc)


class TuneRemote(RemoteTrainBase, AframeRayTask):
    search_space = luigi.Parameter()
    num_samples = luigi.IntParameter()
    min_epochs = luigi.IntParameter()
    max_epochs = luigi.IntParameter()
    reduction_factor = luigi.IntParameter()
    num_workers = luigi.IntParameter(default=ray_worker().replicas)
    gpus_per_worker = luigi.IntParameter(default=ray_worker().gpus_per_replica)

    # image used locally to connect to the ray cluster
    @property
    def default_image(self):
        return "train.sif"

    @property
    def use_wandb(self):
        # always use wandb logging for tune jobs
        return True

    def get_ip(self):
        ip = os.getenv("AFRAME_RAY_CLUSTER_IP")
        ip += ":10001"
        return ip

    def configure_cluster(self, cluster: "KubernetesRayCluster"):
        secret = s3().get_s3_credentials()
        cluster.add_secret("s3-credentials-tune", env=secret)
        cluster.set_env({"AWS_ENDPOINT_URL": s3().get_internal_s3_url()})
        cluster.set_env({"WANDB_API_KEY": wandb().api_key})

        cluster.head["spec"]["template"]["spec"]["containers"][0][
            "imagePullPolicy"
        ] = "Always"
        cluster.worker["spec"]["template"]["spec"]["containers"][0][
            "imagePullPolicy"
        ] = "Always"
        return cluster

    def complete(self):
        # TODO: determine best way of definine
        # completion for tune jobs
        return False

    def run(self):
        from train.tune.cli import main

        args = self.get_args()
        args.append(f"--tune.address={self.get_ip()}")
        args.append(f"--tune.space={self.search_space}")
        args.append(f"--tune.num_workers={self.num_workers}")
        args.append(f"--tune.gpus_per_worker={self.gpus_per_worker}")
        args.append("--tune.cpus_per_gpu=" + str(ray_worker().cpus_per_gpu))
        args.append(f"--tune.num_samples={self.num_samples}")
        args.append(f"--tune.min_epochs={self.min_epochs}")
        args.append(f"--tune.max_epochs={self.max_epochs}")
        args.append(f"--tune.reduction_factor={self.reduction_factor}")
        args.append(f"--tune.storage_dir={self.run_dir}")
        main(args)
