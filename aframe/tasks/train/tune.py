import os
from typing import TYPE_CHECKING

import luigi

from aframe.base import AframeRayTask
from aframe.config import ray_head, ray_worker, s3
from aframe.tasks.train.base import RemoteTrainBase
from aframe.tasks.train.config import wandb

if TYPE_CHECKING:
    from aframe.helm import RayCluster


class TuneRemote(RemoteTrainBase, AframeRayTask):
    name = luigi.Parameter(
        default="ray-tune",
        description="Name of the tune job. "
        "Will be used to group runs in WandB",
    )
    search_space = luigi.Parameter(
        default="train.tune.search_space",
        description="Import path to the search space file "
        "used for hyperparameter tuning. This file is expected "
        "to contain a dictionary named `space` of the search space",
    )
    num_samples = luigi.IntParameter(description="Number of trials to run")
    min_epochs = luigi.IntParameter(
        description="Minimum number of epochs each trial "
        "can run before early stopping is considered."
    )
    max_epochs = luigi.IntParameter(
        description="Maximum number of epochs each trial can run"
    )
    reduction_factor = luigi.IntParameter(
        description="Fraction of poor performing trials to stop early"
    )

    # image used locally to connect to the ray cluster
    @property
    def default_image(self):
        return "train.sif"

    @property
    def use_wandb(self):
        # always use wandb logging for tune jobs
        return True

    def get_ip(self):
        """
        Get the ip of the ray cluster that
        is stored via an environment variable
        """
        ip = os.getenv("AFRAME_RAY_CLUSTER_IP")
        ip += ":10001"
        return ip

    def configure_cluster(self, cluster: "RayCluster"):
        secret = s3().get_s3_credentials()
        values = {
            # image parameters
            "image": self.container,
            "imagePullPolicy": self.image_pull_policy,
            # secrets / env variables
            "awsUrl": s3().get_internal_s3_url(),
            "secret.awsId": secret["AWS_ACCESS_KEY_ID"],
            "secret.awsKey": secret["AWS_SECRET_ACCESS_KEY"],
            "secret.wandbKey": wandb().api_key,
            # compute resources
            "worker.replicas": ray_worker().replicas,
            "worker.cpu": ray_worker().cpus_per_gpu
            * ray_worker().gpus_per_replica,
            "worker.gpu": ray_worker().gpus_per_replica,
            "worker.memory": ray_worker().memory,
            "worker.min_gpu_memory": ray_worker().min_gpu_memory,
            "head.cpu": ray_head().cpus,
            "head.memory": ray_head().memory,
            "dev": str(self.dev).lower(),
        }
        cluster.build_command(values)
        return cluster

    def complete(self):
        # TODO: determine best way of definine
        # completion for tune jobs
        return False

    def run(self):
        from train.tune.cli import main

        args = self.get_args()
        args.append(f"--tune.name={self.name}")
        args.append(f"--tune.address={self.get_ip()}")
        args.append(f"--tune.space={self.search_space}")
        args.append("--tune.num_workers=1")
        args.append("--tune.gpus_per_worker=1")
        args.append(f"--tune.cpus_per_gpu={ray_worker().cpus_per_gpu}")
        args.append(f"--tune.num_samples={self.num_samples}")
        args.append(f"--tune.min_epochs={self.min_epochs}")
        args.append(f"--tune.max_epochs={self.max_epochs}")
        args.append(f"--tune.reduction_factor={self.reduction_factor}")
        args.append(f"--tune.storage_dir={self.run_dir}")
        main(args)
