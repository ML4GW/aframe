import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import jsonargparse
from kr8s.objects import Pod, Secret
from utils.logging import configure_logging

from train.helm import authenticate, setup_kr8s_auth

# Maps external Nautilus S3 URLs to the internal cluster equivalents.
_NAUTILUS_INTERNAL_URLS = {
    "https://s3-west.nrp-nautilus.io": "https://s3-west.nrp-nautilus.io",
    "https://s3-central.nrp-nautilus.io": "http://rook-ceph-rgw-centrals3.rook-central",  # noqa: E501
    "https://s3-east.nrp-nautilus.io": "http://rook-ceph-rgw-easts3.rook-east",
}


@dataclass
class RemoteTrainer:
    """
    Manages a remote training job on a Kubernetes cluster.
    """

    remote_image: str = "ghcr.io/ml4gw/aframe/train:main"
    request_gpus: int = 8
    cpus_per_gpu: int = 12
    memory_per_cpu: float = 4.0
    base_memory: int = 32
    min_gpu_memory: int = 15000
    aws_endpoint_url: str = field(
        default_factory=lambda: os.getenv("AWS_ENDPOINT_URL", "")
    )
    aws_access_key_id: str = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    aws_secret_access_key: str = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    wandb_api_key: str = field(
        default_factory=lambda: os.getenv("WANDB_API_KEY", "")
    )
    use_git_sync: bool = False
    git_url: str = "git@github.com:ML4GW/aframe.git"
    git_ref: str = "main"
    namespace: str = "bbhnet"
    pod_creation_timeout: int = 7200

    @property
    def internal_endpoint_url(self) -> str:
        """
        Remap external Nautilus URLs to their internal cluster equivalents.
        """
        return _NAUTILUS_INTERNAL_URLS.get(
            self.aws_endpoint_url, self.aws_endpoint_url
        )

    @property
    def num_cpus(self) -> int:
        return self.request_gpus * self.cpus_per_gpu

    @property
    def cpu_memory(self) -> str:
        memory = self.memory_per_cpu * self.num_cpus + self.base_memory
        return f"{memory}G"

    def git_secret(self) -> Secret:
        ssh_key_path = Path.home() / ".ssh" / "id_rsa"
        with open(ssh_key_path) as f:
            key = f.read()
        encoded = base64.b64encode(key.encode("ascii")).decode("ascii")
        spec = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "git-creds", "namespace": self.namespace},
            "type": "Opaque",
            "data": {"ssh": encoded},
        }
        return Secret(resource=spec)

    def pod_spec(self, args: list[str]) -> dict:
        spec = {
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
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
                }
            },
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "train",
                    "image": self.remote_image,
                    "imagePullPolicy": "Always",
                    "command": ["python", "-m", "train"],
                    "args": args,
                    "volumeMounts": [
                        {"mountPath": "/dev/shm", "name": "dshm"},
                    ],
                    "resources": {
                        "limits": {
                            "memory": self.cpu_memory,
                            "cpu": f"{self.num_cpus}",
                            "nvidia.com/gpu": f"{self.request_gpus}",
                        },
                        "requests": {
                            "memory": self.cpu_memory,
                            "cpu": f"{self.num_cpus}",
                            "nvidia.com/gpu": f"{self.request_gpus}",
                        },
                    },
                    "envFrom": [{"secretRef": {"name": "s3-credentials"}}],
                    "env": [
                        {
                            "name": "AWS_ENDPOINT_URL",
                            "value": self.internal_endpoint_url,
                        },
                        {
                            "name": "WANDB_API_KEY",
                            "value": self.wandb_api_key,
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "dshm",
                    "emptyDir": {"sizeLimit": "256Gi", "medium": "Memory"},
                },
            ],
        }

        if self.use_git_sync:
            spec["initContainers"] = [
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
                        {"name": "aframe", "mountPath": "/opt"},
                        {
                            "name": "git-creds",
                            "mountPath": "/etc/git-secret",
                            "readOnly": True,
                        },
                    ],
                    "securityContext": {"runAsUser": 65533},
                }
            ]
            spec["containers"][0]["volumeMounts"].append(
                {"mountPath": "/opt", "name": "aframe"}
            )
            spec["volumes"] += [
                {"name": "aframe", "emptyDir": {}},
                {"name": "git-creds", "secret": {"secretName": "git-creds"}},
            ]

        return spec

    def wait_for_completion(self, pod: Pod) -> str:
        """
        Poll the pod until it terminates.
        """
        start = time.monotonic()
        while True:
            pod.refresh()
            phase = pod.status.phase
            if phase in ("Succeeded", "Failed"):
                return phase
            elapsed = time.monotonic() - start
            if elapsed > self.pod_creation_timeout and phase == "Pending":
                raise TimeoutError(
                    f"Pod still Pending after {self.pod_creation_timeout}s"
                )
            time.sleep(10)

    def run(self, args: list[str]) -> None:
        """
        Launch a training pod on Kubernetes, stream its logs, and
        wait for completion. Cleans up secrets and the pod on exit.

        Args:
            args: Arguments forwarded verbatim to `python -m train`
                  inside the pod (e.g. ["fit", "--config", "..."]).
        """
        authenticate()
        setup_kr8s_auth()

        s3_secret = Secret(
            resource={
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "s3-credentials",
                    "namespace": self.namespace,
                },
                "type": "Opaque",
                "stringData": {
                    "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
                    "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
                },
            }
        )
        git_secret = self.git_secret() if self.use_git_sync else None

        if not s3_secret.exists():
            s3_secret.create()
        if git_secret is not None and not git_secret.exists():
            git_secret.create()

        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "aframe-train",
                "namespace": self.namespace,
            },
            "spec": self.pod_spec(args),
        }
        pod = Pod(resource=pod_manifest)

        try:
            pod.create()
            logging.info(
                f"Created pod aframe-train in namespace {self.namespace}"
            )
            for line in pod.logs(follow=True):
                logging.info(line)
            phase = self.wait_for_completion(pod)
            if phase != "Succeeded":
                raise RuntimeError(
                    f"Training pod terminated with phase '{phase}'"
                )
            logging.info("Training completed successfully")
        finally:
            pod.delete()
            s3_secret.delete()
            if git_secret is not None:
                git_secret.delete()


def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_class_arguments(RemoteTrainer)
    parser.add_argument(
        "--train_args",
        nargs="*",
        default=[],
        help="Arguments forwarded verbatim to `python -m train` in the pod "
        "(e.g. fit --config config.yaml --data.sample_rate 2048).",
    )
    return parser


def main(args=None):
    parser = build_parser()
    cfg = parser.parse_args(args)

    if cfg.logfile is not None:
        Path(cfg.logfile).parent.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg.logfile, cfg.verbose)

    cfg = parser.instantiate_classes(cfg)
    cfg.trainer.run(cfg.train_args)


if __name__ == "__main__":
    main()
