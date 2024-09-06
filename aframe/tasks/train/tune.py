import os
from typing import TYPE_CHECKING

import luigi

from aframe.base import AframeRayTask
from aframe.config import ray_head, ray_worker, s3, ssh, wandb
from aframe.targets import LawS3Target
from aframe.tasks.train.base import RemoteTrainBase

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
    workers_per_trial = luigi.IntParameter(
        default=1, description="Number of ray workers to use per trial"
    )
    gpus_per_worker = luigi.IntParameter(
        default=1, description="Number of gpus to allocate to each ray worker"
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
        return f"ray://{ip}"

    def configure_cluster(self, cluster: "RayCluster"):
        # get ssh key for git-sync init container
        with open(ssh().ssh_file, "r") as f:
            ssh_key = f.read()

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
            "worker.cpu": ray_worker().cpus_per_replica,
            "worker.gpu": ray_worker().gpus_per_replica,
            "worker.memory": ray_worker().memory_per_replica,
            "worker.min_gpu_memory": ray_worker().min_gpu_memory,
            "head.cpu": ray_head().cpus,
            "head.memory": ray_head().memory,
            "dev": str(self.dev).lower(),
            "gitRepo.sshKey": ssh_key,
        }
        cluster.build_command(values)
        return cluster

    def output(self):
        path = self.run_dir / "best.pt"
        return LawS3Target(str(path))

    def run(self):
        from lightray.tune import run
        from ray.tune.schedulers import ASHAScheduler

        from train.callbacks import TraceModel
        from train.cli import AframeCLI

        args = self.get_args()

        scheduler = ASHAScheduler(
            max_t=self.max_epochs,
            grace_period=self.min_epochs,
            reduction_factor=self.reduction_factor,
        )

        metric_name = "valid_auroc"
        objective = "max"
        prefix = "s3://" if str(self.run_dir).startswith("s3://") else ""
        results = run(
            cli_cls=AframeCLI,
            name=self.name,
            scheduler=scheduler,
            metric_name=metric_name,
            objective=objective,
            search_space=self.search_space,
            num_samples=self.num_samples,
            workers_per_trial=self.workers_per_trial,
            gpus_per_worker=self.gpus_per_worker,
            cpus_per_gpu=ray_worker().cpus_per_gpu,
            storage_dir=self.run_dir,
            callbacks=[TraceModel],
            address=self.get_ip(),
            args=args,
        )
        # return path to best model weights from best trial
        best = results.get_best_result(
            metric=metric_name, mode=objective, scope="all"
        )
        best = best.get_best_checkpoint(metric=metric_name, mode=objective)
        weights = os.path.join(prefix, best.path, "model.pt")

        # copy the best weights to the output location
        s3().client.copy(weights, self.output().path)
