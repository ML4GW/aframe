import os
import shlex
import sys

import law
import luigi

from aframe.base import AframeTask, logger
from aframe.config import Defaults
from aframe.utils import stream_command


class TrainBase(AframeTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    config = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=101588)
    use_wandb = luigi.BoolParameter()

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

    def get_args(self):
        args = [
            "--config",
            self.config,
            "--seed_everything",
            str(self.seed),
            f"--data.ifos=[{','.join(self.cfg.ifos)}]",
            "--data.data_dir",
            self.data_dir,
        ]
        if self.use_wandb and not self.cfg.wandb.api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key. "
                "Try setting the WANDB_API_KEY environment variable."
            )
        elif self.use_wandb:
            args = self.configure_wandb(args)
        args.append(f"--trainer.logger.save_dir={self.run_dir}")
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
        dir = law.LocalDirectoryTarget(self.run_dir)
        return dir.child("model.pt", type="f")
