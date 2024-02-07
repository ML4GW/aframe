import os
from typing import TYPE_CHECKING

import luigi

from aframe.base import AframeRayTask
from aframe.config import ray_worker, s3
from aframe.tasks.train.base import RemoteTrainBase
from aframe.tasks.train.config import wandb

if TYPE_CHECKING:
    from ray_kube import KubernetesRayCluster


class TuneRemote(RemoteTrainBase, AframeRayTask):
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
    cpus_per_gpu = luigi.IntParameter(
        default=8, description="Number of CPUs to allocate to each gpu"
    )

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
        """
        Get the ip of the ray cluster that
        is stored via an environment variable
        """
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
        args.append(f"--tune.cpus_per_gpu={self.cpus_per_gpu}")
        args.append(f"--tune.num_samples={self.num_samples}")
        args.append(f"--tune.min_epochs={self.min_epochs}")
        args.append(f"--tune.max_epochs={self.max_epochs}")
        args.append(f"--tune.reduction_factor={self.reduction_factor}")
        args.append(f"--tune.storage_dir={self.run_dir}")
        main(args)
