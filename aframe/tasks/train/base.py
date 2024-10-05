from typing import List

import law
import luigi
from luigi.util import inherits

from aframe.config import Defaults, paths, wandb
from aframe.parameters import PathParameter
from aframe.tasks.data import TrainingWaveforms, ValidationWaveforms
from aframe.tasks.data.fetch import FetchTrain


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
    sample_rate = luigi.FloatParameter(
        description="Sample rate of the data in Hz"
    )
    highpass = luigi.FloatParameter(
        description="Highpass frequency in Hz to apply during whitening"
    )
    fftlength = luigi.FloatParameter(
        description="Duration in seconds of data used for FFT calculation"
    )
    q = luigi.OptionalFloatParameter(
        default=None, description="Value of Q used for Q-transform"
    )
    fduration = luigi.FloatParameter(
        description="Duration in seconds of the whitening filter to use,"
    )
    run_dir = PathParameter(
        description="Directory where the training logger "
        "will save checkpoints, logs, etc. ",
        default=paths().train_rundir,
    )
    data_dir = PathParameter(
        description="Directory where training data is stored."
        "It is expected to contain a `signals.hdf5` file of signals, "
        "and a `/background` sub-directory containing background "
        "files used for training",
        default=paths().train_datadir,
    )
    ckpt_path = PathParameter(
        default="",
        description="Path to checkpoint file from which "
        "to restart training",
    )


@inherits(TrainBaseParameters)
class TrainBase(law.Task):
    def requires(self):
        reqs = {}
        reqs["strain"] = FetchTrain.req(self)
        reqs["train_waveforms"] = TrainingWaveforms.req(self)

        reqs["val_waveforms"] = ValidationWaveforms.req(self)
        return reqs

    def configure_wandb(self, args: List[str]) -> None:
        # note that we append the wandb logger
        # so that we always use csv file to store metrics
        args.append("--trainer.logger+=WandbLogger")
        args.append("--trainer.logger.job_type=train")
        args.append(f"--trainer.logger.save_dir={self.run_dir}")

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
        args.append("--data.sample_rate=" + str(self.sample_rate))
        args.append("--data.kernel_length=" + str(self.kernel_length))
        args.append("--data.fduration=" + str(self.fduration))
        args.append("--data.fftlength=" + str(self.fftlength))
        args.append("--data.highpass=" + str(self.highpass))
        if self.q is not None:
            args.append("--data.q=" + str(self.q))
        return args

    def get_args(self):
        args = [
            "--config",
            self.config,
            "--seed_everything",
            str(self.seed),
            f"--data.ifos=[{','.join(self.ifos)}]",
            "--data.data_dir",
            str(self.data_dir),
        ]
        args = self.configure_data_args(args)
        if self.use_wandb and not wandb().api_key:
            raise ValueError(
                "Can't run W&B experiment without specifying an API key. "
                "Try setting the WANDB_API_KEY environment variable."
            )

        # first, set the save_dir of the CSV logger
        args.append(f"--trainer.logger.save_dir={self.run_dir}")
        if self.use_wandb:
            args = self.configure_wandb(args)

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
        default=12, description="Number of CPUs to request per GPU"
    )
    memory_per_cpu = luigi.FloatParameter(
        default=4, description="Amount of memory to request per CPU in GB"
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
