import os
from typing import List

import law
import luigi
from luigi.util import inherits

from aframe.config import Defaults
from aframe.tasks.data import TrainWaveforms, ValidationWaveforms
from aframe.tasks.data.fetch import FetchTrain
from aframe.tasks.train.config import wandb


class TrainBaseParameters(law.Task):
    config = luigi.Parameter(
        default=Defaults.TRAIN,
        description="Path to the lightning CLI config file used for training. "
        "Defaults to the config file in the root of the train project.",
    )
    ifos = luigi.ListParameter(
        default=["H1", "L1"], description="List of ifos to use for training."
    )
    seed = luigi.IntParameter(
        default=101588,
        description="Integer seed value used to seed training run",
    )
    use_wandb = luigi.BoolParameter(
        default=False, description="Whether to use W&B for logging"
    )
    kernel_length = luigi.FloatParameter(
        description="Length of the kernel in seconds "
        "the neural-network will analyze"
    )
    highpass = luigi.FloatParameter(
        description="Highpass frequency in Hz to apply during whitening"
    )
    fduration = luigi.FloatParameter(
        description="Duration in seconds of the whitening filter to use,"
    )
    run_dir = luigi.Parameter(
        description="Directory where the training logger "
        "will save checkpoints, logs, etc. "
    )
    data_dir = luigi.Parameter(
        description="Directory where training data is stored."
        "It is expected to contain a `signals.hdf5` file of signals, "
        "and a `/background` sub-directory containing background "
        "files used for training"
    )


@inherits(TrainBaseParameters)
class TrainBase(law.Task):
    def requires(self):
        reqs = {}
        reqs["strain"] = FetchTrain.req(
            self,
            segments_file=os.path.join(self.data_dir, "segments.txt"),
            data_dir=os.path.join(self.data_dir, "background"),
        )
        reqs["train_waveforms"] = TrainWaveforms.req(
            self,
            output_file=os.path.join(self.data_dir, "train_waveforms.hdf5"),
        )

        reqs["val_waveforms"] = ValidationWaveforms.req(
            self,
            output_dir=self.data_dir,
        )
        return reqs

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


class RemoteParameters(law.Task):
    remote_image = luigi.Parameter(
        default="ghcr.io/ml4gw/aframev2/train:main",
        description="The container image to use for "
        "training on the nautilus cluster ",
    )
    min_gpu_memory = luigi.IntParameter(
        default=15000, description="Minimum amount of memory per GPU in MB"
    )
    request_gpus = luigi.IntParameter(
        default=8,
        description="Number of GPUs to request for training",
    )
    cpus_per_gpu = luigi.IntParameter(
        default=8, description="Number of CPUs to request per GPU"
    )
    memory_per_cpu = luigi.FloatParameter(
        default=1.5, description="Amount of memory to request per CPU in GB"
    )

    @property
    def num_cpus(self):
        """
        Total number of CPUs to request for training
        """
        return self.request_gpus * self.cpus_per_gpu

    @property
    def base_memory(self):
        """
        Base CPU memory to request for training
        """
        return 32

    @property
    def cpu_memory(self):
        """
        Total CPU memory to request for training
        """
        memory = self.memory_per_cpu * self.num_cpus
        memory += self.base_memory
        return f"{memory}G"


class RemoteTrainBase(TrainBase, RemoteParameters):
    pass
