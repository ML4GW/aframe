from typing import List

import law
import luigi
from luigi.util import inherits

from aframe.config import Defaults
from aframe.tasks.train.config import wandb


class TrainBaseParameters(law.Task):
    config = luigi.Parameter(default=Defaults.TRAIN)
    ifos = luigi.ListParameter(default=["H1", "L1"])
    data_dir = luigi.Parameter()
    run_dir = luigi.Parameter()
    seed = luigi.IntParameter(default=101588)
    use_wandb = luigi.BoolParameter(default=False)
    kernel_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    fduration = luigi.FloatParameter()


class RemoteParameters(law.Task):
    image = luigi.Parameter(default="ghcr.io/ml4gw/aframev2/train:main")
    min_gpu_memory = luigi.IntParameter(default=15000)
    request_gpus = luigi.IntParameter(default=1)
    request_cpus = luigi.IntParameter(default=1)
    request_cpu_memory = luigi.Parameter("4G")


class TrainParameters(TrainBaseParameters, RemoteParameters):
    pass


@inherits(TrainBaseParameters)
class TrainBase(law.Task):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.data_dir:
            raise ValueError("Must specify data directory")
        if not self.run_dir:
            raise ValueError("Must specify run root directory")

    def configure_wandb(self, args: List[str]) -> None:
        args.append("--trainer.logger=WandbLogger")
        args.append("--trainer.logger.job_type=train")

        for key in ["name", "entity", "project", "group", "tags"]:
            value = getattr(wandb(), key)
            if value and key != "tags":
                args.append(f"--trainer.logger.{key}={value}")
            elif value:
                for v in value.split(","):
                    args.append(f"--trainer.logger.{key}+={v}")
        return args

    def configure_data_args(self, args: List[str]) -> None:
        # configure ONLY the data arguments that
        # are shared by multiple tasks so that we can specify them
        # in one place (the aframe config) and have them overwrite
        # the defaults set in the train projects config.yaml.
        # Arguments that are only used in the train project
        # (e.g. model architecture config, training hyperparams,
        # logging, profiling, etc.) should be specified in the
        # train projects config.yaml file.
        args.append("--data.kernel_length=" + str(self.kernel_length))
        args.append("--data.fduration=" + str(self.fduration))
        args.append("--data.highpass=" + str(self.highpass))
        return args

    def get_args(self):
        args = [
            "--config",
            self.config,
            "--seed_everything",
            str(self.seed),
            f"--data.ifos=[{','.join(self.ifos)}]",
            "--data.data_dir",
            self.data_dir,
        ]
        args = self.configure_data_args(args)
        if self.use_wandb and not wandb().api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key. "
                "Try setting the WANDB_API_KEY environment variable."
            )
        # wandb logger uses save_dir, csv logger uses log_dir :(
        elif self.use_wandb:
            args = self.configure_wandb(args)

        args.append(f"--trainer.logger.save_dir={self.run_dir}")

        return args

    def run(self):
        raise NotImplementedError


@inherits(RemoteParameters)
class RemoteTrainBase(TrainBase):
    pass
