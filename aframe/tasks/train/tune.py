import os
from typing import TYPE_CHECKING

import law
import luigi
from luigi.util import inherits

from aframe.base import AframeRayTask, AframeSingularityTask, AframeWrapperTask
from aframe.config import ray_head, ray_worker, s3, ssh, wandb
from aframe.targets import Bytes, LawS3Target
from aframe.tasks.train.base import RemoteTrainBase, TrainBase

if TYPE_CHECKING:
    from aframe.helm import RayCluster


class TuneLocal(TrainBase, AframeSingularityTask):
    tune_config = luigi.Parameter(
        description="Path to the `yaml` file used"
        " to configure the lightray tune job. "
    )

    @property
    def default_image(self):
        return "train.sif"

    def output(self):
        path = self.run_dir / "best.pt"
        return law.LocalFileTarget(str(path), format=Bytes)

    def run(self):
        from lightray.cli import cli

        args = ["--config", self.tune_config, "--"]
        lightning_args = self.get_args()
        lightning_args.pop(
            0
        )  # remove "fit" subcommand since lightray takes care of it
        args.extend(lightning_args)

        results = cli(args)
        prefix = "s3://" if str(self.run_dir).startswith("s3://") else ""

        # return path to best model weights from best trial
        best = results.get_best_result(scope="all")
        best = best.get_best_checkpoint()
        weights = os.path.join(prefix, best.path, "model.pt")

        # copy the best weights to the output location
        s3().client.copy(weights, self.output().path)


class TuneRemote(RemoteTrainBase, AframeRayTask):
    git_url = luigi.Parameter(
        default="git@github.com:ML4GW/aframev2.git",
        description="Git repository url to clone and"
        " mount into the kubernetes pod. "
        "Only used if `dev` is set to True",
    )
    git_ref = luigi.Parameter(
        default="main",
        description="Git branch or commit to checkout. "
        "Only used if `dev` is set to True",
    )
    tune_config = luigi.Parameter(
        description="Path to the `yaml` file used"
        " to configure the lightray tune job. "
    )

    # image used locally to connect to the ray cluster
    @property
    def default_image(self):
        return "train.sif"

    @property
    def use_wandb(self):
        # always use wandb logging for tune jobs
        return True

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
            "gitRepo.url": self.git_url,
            "gitRepo.ref": self.git_ref,
        }
        cluster.build_command(values)
        return cluster

    def output(self):
        path = self.run_dir / "best.pt"
        return LawS3Target(str(path), format=Bytes)

    def run(self):
        from lightray.cli import cli

        args = ["--config", self.tune_config, "--"]
        args.extend(self.get_args())

        results = cli(args)
        prefix = "s3://" if str(self.run_dir).startswith("s3://") else ""

        # return path to best model weights from best trial
        best = results.get_best_result(scope="all")
        best = best.get_best_checkpoint()
        weights = os.path.join(prefix, best.path, "model.pt")

        # copy the best weights to the output location
        s3().client.copy(weights, self.output().path)


@inherits(TuneLocal, TuneRemote)
class Tune(AframeWrapperTask):
    """
    Class that dynamically chooses between
    remote training on nautilus or local training on LDG.

    Useful for incorporating into pipelines where
    you don't care where the training is run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remote = self.validate_dirs()

    def validate_dirs(self) -> bool:
        # train remotely if run_dir stars with s3://

        # Note: one can specify a remote data_dir, but
        # train locally
        remote = str(self.run_dir).startswith("s3://")

        if remote and not str(self.data_dir).startswith("s3://"):
            raise ValueError(
                "If run_dir is an s3 path, data_dir must also be an s3 path"
                "Got data_dir: {self.data_dir} and run_dir: {self.run_dir}"
            )
        return remote

    def requires(self):
        if self.remote:
            return TuneRemote.req(self)
        else:
            return TuneLocal.req(self)
