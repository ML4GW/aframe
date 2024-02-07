import json
import os
import shlex
import sys
from typing import Dict

import law
import luigi
import yaml
from kr8s.objects import Secret
from luigi.contrib.kubernetes import KubernetesJobTask
from luigi.util import inherits

from aframe.base import AframeSingularityTask, AframeWrapperTask, logger
from aframe.config import s3
from aframe.targets import Bytes, LawS3Target
from aframe.tasks.train.base import (
    RemoteParameters,
    RemoteTrainBase,
    TrainBase,
    TrainBaseParameters,
)
from aframe.tasks.train.config import wandb
from aframe.utils import stream_command


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
    dev = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.run_dir.startswith("s3://"):
            raise ValueError(
                "run_dir must be an s3 path for remote training tasks"
            )
        if not self.data_dir.startswith("s3://"):
            raise ValueError(
                "data_dir must be an s3 path for remote training tasks"
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
        return 7200

    def get_config(self):
        with open(self.config, "r") as f:
            doc = yaml.safe_load(f)
            json_string = json.dumps(doc)
        return json_string

    def get_args(self):
        # get args from base class removing the first two
        # which reference the config file. We'll
        # need to set this as an environment variable of
        # raw yaml content to run remotely.
        args = super().get_args()
        args = args[2:]
        args = ["--config", self.get_config()] + args
        return args

    def output(self):
        return LawS3Target(
            os.path.join(self.run_dir, "model.pt"), format=Bytes
        )

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
                "image": self.remote_image,
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
                "imagePullPolicy": "Always",
                "command": ["python", "-m", "train"],
                "args": self.get_args(),
                "resources": {
                    "limits": {
                        "memory": f"{self.cpu_memory}",
                        "cpu": f"{self.num_cpus}",
                        "nvidia.com/gpu": f"{self.request_gpus}",
                    },
                    "requests": {
                        "memory": f"{self.cpu_memory}",
                        "cpu": f"{self.num_cpus}",
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
        if not self.secret.exists():
            self.secret.create()
        super().run()

    def on_failure(self, exc):
        self.secret.delete()
        super().on_failure(exc)

    def on_success(self):
        self.secret.delete()
        super().on_success()


@inherits(TrainBaseParameters, RemoteParameters)
class Train(AframeWrapperTask):
    """
    Class that dynamically chooses between
    remote training on nautilus or local training on LDG.

    Useful for incorporating into pipelines where
    you don't care where the training is run.
    """

    train_remote = luigi.BoolParameter(
        default=False,
        description="If `True`, run training remotely on nautilus"
        " otherwise run locally",
    )

    def requires(self):
        if self.train_remote:
            return TrainRemote.req(self)
        else:
            return TrainLocal.req(self)
