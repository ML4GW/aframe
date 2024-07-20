import base64
import json
import shlex
import sys
from pathlib import Path
from typing import Dict

import law
import luigi
import yaml
from kr8s.objects import Secret
from luigi.contrib.kubernetes import KubernetesJobTask
from luigi.util import inherits

from aframe.base import AframeSingularityTask, AframeWrapperTask, logger
from aframe.config import s3, wandb
from aframe.targets import Bytes, LawS3Target
from aframe.tasks.train.base import RemoteTrainBase, TrainBase
from aframe.tasks.train.utils import stream_command


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
        dir = law.LocalDirectoryTarget(str(self.run_dir))
        return dir.child("model.pt", type="f")


class TrainRemote(KubernetesJobTask, RemoteTrainBase):
    dev = luigi.BoolParameter(default=False)
    use_init_container = luigi.BoolParameter(
        default=False,
        description="Whether to use the git-sync init-container to sync "
        "a remote aframe git repository into the pod Defaults to False, "
        "in which case the code added to the container image at build "
        "time will be used",
    )
    git_url = luigi.Parameter(
        default="git@github.com:ML4GW/aframev2.git",
        description="The git repository to clone into the pod"
        "Only relevant if `use_init_container` is True",
    )
    git_ref = luigi.Parameter(
        default="main",
        description="The git branch or commit to checkout"
        "Only relevant if `use_init_container` is True",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not str(self.run_dir).startswith("s3://"):
            raise ValueError(
                "run_dir must be an s3 path for remote training tasks"
            )
        if not str(self.data_dir).startswith("s3://"):
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
        # read in training config into a json string
        # to pass to the remote training job via
        # the jsonargparse command line
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
        return LawS3Target(str(self.run_dir / "model.pt"), format=Bytes)

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
    def s3_secret(self):
        # kubernetes config for creating
        # secret containing credentials for s3 access
        spec = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "s3-credentials", "type": "Opaque"},
        }
        spec["stringData"] = s3().get_s3_credentials()

        return Secret(resource=spec)

    def git_secret(self):
        # kubernetes config for creating
        # secret containing users ssh
        # key for git access if using git-sync init containers
        spec = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "git-creds", "type": "Opaque"},
        }
        ssh_key = Path.home() / ".ssh" / "id_rsa"
        with open(ssh_key, "r") as f:
            key = f.read()

        ssh_key = base64.b64encode(key.encode("ascii")).decode("ascii")

        spec["data"] = {"ssh": ssh_key}

        return Secret(resource=spec)

    @property
    def init_containers(self):
        # kubernetes config for creating
        # init container to sync a remote git
        # repository into the pod at run time
        config = [
            {
                "name": "git-sync",
                "image": "registry.k8s.io/git-sync/git-sync:v4.2.1",
                "env": [
                    {"name": "GITSYNC_REPO", "value": self.git_url},
                    {"name": "GITSYNC_REF", "value": self.git_ref},
                    {"name": "GITSYNC_ROOT", "value": "/opt"},
                    {"name": "GITSYNC_LINK", "value": "aframe"},
                    {"name": "GITSYNC_ONE_TIME", "value": "true"},
                    {"name": "GITSYNC_SSH_KNOWN_HOSTS", "value": "false"},
                    {"name": "GITSYNC_SUBMODULES", "value": "recursive"},
                    {"name": "GITSYNC_ADD_USER", "value": "true"},
                    {"name": "GITSYNC_SYNC_TIMEOUT", "value": "360s"},
                ],
                "volumeMounts": [
                    {"name": self.name, "mountPath": "/opt"},
                    {
                        "name": f"{self.name}-git-secret",
                        "mountPath": "/etc/git-secret",
                        "readOnly": True,
                    },
                ],
                "securityContext": {"runAsUser": 65533},
            }
        ]

        return config

    @property
    def spec_schema(self):
        spec = self.gpu_constraints
        spec["containers"] = [
            {
                "name": "train",
                "image": self.remote_image,
                "volumeMounts": [
                    {"mountPath": "/dev/shm", "name": "dshm"},
                    {"mountPath": "/opt", "name": self.name},
                ],
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

        if self.use_init_container:
            spec["initContainers"] = self.init_containers

        spec["volumes"] = [
            {
                "name": "dshm",
                "emptyDir": {"sizeLimit": "128Gi", "medium": "Memory"},
            },
        ]
        if self.use_init_container:
            spec["volumes"] += [
                {"name": self.name, "emptyDir": {}},
                {
                    "name": f"{self.name}-git-secret",
                    "secret": {"secretName": "git-creds"},
                },
            ]
        return spec

    def run(self):
        if not self.s3_secret.exists():
            self.s3_secret.create()

        if self.use_init_container and not self.git_secret().exists():
            self.git_secret().create()
        super().run()

    def on_failure(self, exc):
        self.s3_secret.delete()
        if self.use_init_container:
            self.git_secret().delete()
        super().on_failure(exc)

    def on_success(self):
        self.s3_secret.delete()
        if self.use_init_container:
            self.git_secret().delete()
        super().on_success()


@inherits(TrainLocal, TrainRemote)
class Train(AframeWrapperTask):
    """
    Class that dynamically chooses between
    remote training on nautilus or local training on LDG.

    Useful for incorporating into pipelines where
    you don't care where the training is run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_remote = self.validate_dirs()

    def validate_dirs(self) -> bool:
        # train remotely if run_dir stars with s3://

        # Note: one can specify a remote data_dir, but
        # train locally
        train_remote = str(self.run_dir).startswith("s3://")

        if train_remote and not str(self.data_dir).startswith("s3://"):
            raise ValueError(
                "If run_dir is an s3 path, data_dir must also be an s3 path"
                "Got data_dir: {self.data_dir} and run_dir: {self.run_dir}"
            )
        return train_remote

    def requires(self):
        if self.train_remote:
            return TrainRemote.req(self)
        else:
            return TrainLocal.req(self)
