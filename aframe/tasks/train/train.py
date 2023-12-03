import shlex
import sys
from configparser import ConfigParser

import law
import luigi

from aframe.base import AframeRayTask, logger
from aframe.config import ray_worker
from aframe.tasks.train.base import TrainBase
from aframe.tasks.train.config import s3, wandb
from aframe.utils import stream_command


class TrainLocal(TrainBase):
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


class TuneRemote(TrainBase, AframeRayTask):
    search_space = luigi.Parameter()
    num_samples = luigi.IntParameter()
    min_epochs = luigi.IntParameter()
    max_epochs = luigi.IntParameter()
    reduction_factor = luigi.IntParameter()

    def configure_cluster(self, cluster):
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

        # TODO: add AWS_ENDPOINT_URL to cluster environment

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
