import os
import shlex
import sys
from configparser import ConfigParser

import law
import luigi

from aframe.base import AframeRayTask, AframeTask, logger
from aframe.config import Defaults
from aframe.utils import stream_command


class TrainBase(AframeTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    config = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=101588)
    use_wandb = luigi.BoolParameter()
    profile = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.data_dir:
            raise ValueError("Must specify data directory")
        if not self.run_dir:
            raise ValueError("Must specify run root directory")
        self.config = self.config or Defaults.TRAIN

    def configure_wandb(self, args: list[str]) -> None:
        args.append("--trainer.logger=WandbLogger")
        args.append("--trainer.logger.job_type=train")

        for key in ["name", "entity", "project", "group", "tags"]:
            value = getattr(self.cfg.wandb, key)
            if value and key != "tags":
                args.append(f"--trainer.logger.{key}={value}")
            elif value:
                for v in value.split(","):
                    args.append(f"--trainer.logger.{key}+={v}")
        return args
    
    def configure_profiler(self, args: list[str]) -> None:
        args.append("--trainer.profiler=PytorchProfiler")
        args.append("--trainer.profiler.profile_memory")
        return args

    def get_args(self):
        args = [
            "--config",
            self.config,
            "--seed_everything",
            str(self.seed),
            f"--data.ifos=[{','.join(self.cfg.train.ifos)}]",
            "--data.data_dir",
            self.data_dir,
        ]
        if self.use_wandb and not self.cfg.wandb.api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key. "
                "Try setting the WANDB_API_KEY environment variable."
            )
        # wandb logger uses save_dir, csv logger uses log_dir :(
        elif self.use_wandb:
            args = self.configure_wandb(args)
            args.append(f"--trainer.logger.log_dir={self.run_dir}")
        else:
            args.append(f"--trainer.logger.save_dir={self.run_dir}")

        args.append(f"--trainer.logger.name=train_logs")
        if self.profile:
            args = self.configure_profiler(args)
        
        return args

    def run(self):
        raise NotImplementedError


class TrainLocal(TrainBase):
    def sandbox_env(self, _) -> dict[str, str]:
        env = super().sandbox_env(_)
        if self.cfg.wandb.api_key:
            env["WANDB_API_KEY"] = self.cfg.wandb.api_key
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
        dir = law.LocalDirectoryTarget(os.path.join(self.run_dir, "train_logs", "version_0"))
        return dir.child("model.pt", type="f")


class TuneRemote(TrainBase, AframeRayTask):
    search_space = luigi.Parameter()
    num_samples = luigi.IntParameter()
    min_epochs = luigi.IntParameter()
    max_epochs = luigi.IntParameter()
    reduction_factor = luigi.IntParameter()

    def configure_cluster(self, cluster):
        config = ConfigParser.read(self.cfg.s3.credentials)
        keys = ["aws_access_key_id", "aws_secret_access_key"]
        secret = {}
        for key in keys:
            try:
                value = config["default"][key]
            except KeyError:
                raise ValueError(
                    "aws credentials file {} is missing "
                    "key {} in default table".format(
                        self.cfg.s3.credentials, key
                    )
                )
            secret[key] = value
        cluster.add_secret("s3-credentials", env=secret)

        # TODO: add AWS_ENDPOINT_URL to cluster environment

    def run(self):
        from train.tune import cli as main

        args = self.get_args()
        args.append(f"--tune.num_workers={self.cfg.ray_worker.replicas}")
        args.append(
            "--tune.gpus_per_worker="
            + str(self.cfg.ray_worker.gpus_per_worker)
        )
        args.append(
            "--tune.cpus_per_gpu=" + str(self.cfg.ray_worker.cpus_per_gpu)
        )
        args.append("--tune.num_samples={self.num_samples}")
        args.append("--tune.min_epochs={self.min_epochs}")
        args.append("--tune.max_epochs={self.max_epochs}")
        args.append("--tune.reduction_factor={self.reduction_factor}")
        args.append("--tune.storage_dir={self.run_dir}/ray")

        main(args)
