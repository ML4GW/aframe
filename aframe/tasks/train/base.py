from typing import List

import law
import luigi
from luigi.util import inherits

from aframe.config import Defaults, paths, wandb
from aframe.parameters import PathParameter
from aframe.tasks.data import TrainingWaveforms, ValidationWaveforms
from aframe.tasks.data.fetch import FetchTrain


class TrainBaseParameters(law.Task):
    train_config = luigi.Parameter(
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
    lowpass = luigi.OptionalFloatParameter(
        description="Lowpass frequency in Hz to apply during whitening",
        default="",
    )
    fftlength = luigi.OptionalFloatParameter(
        description="Duration in seconds of data used for FFT calculation",
        default="",
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
    background_dir = PathParameter(
        description="Directory where training background is stored."
        "It is expected to contain a set of hdf5 files with names "
        "`background-<gps_start_time>_<duration>.hdf5` containing "
        "strain data.",
        default=paths().train_background_dir,
    )
    waveforms_dir = PathParameter(
        description="Directory where training waveforms are stored."
        "It is expected to contain a `val_waveforms.hdf5` file of "
        "validation signals and a `training_waveforms.hdf5` file containing "
        "training signals if `precompute_train_waveforms` is set to True.",
        default=paths().train_waveforms_dir,
    )
    precompute_train_waveforms = luigi.BoolParameter(
        default=False,
        description="Whether to pre-compute the waveforms used "
        "during training. If True, the training waveforms will be "
        "read from the `training_waveforms.hdf5` file in the data "
        "directory. If False, the waveforms will be simulated "
        "on-the-fly during training.",
    )


@inherits(TrainBaseParameters)
class TrainBase(law.Task):
    def requires(self):
        reqs = {}
        reqs["strain"] = FetchTrain.req(self)
        reqs["val_waveforms"] = ValidationWaveforms.req(self)
        if self.precompute_train_waveforms:
            reqs["train_waveforms"] = TrainingWaveforms.req(self)
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
        fftlength = self.fftlength or "null"
        lowpass = self.lowpass or "null"
        args.append("--data.sample_rate=" + str(self.sample_rate))
        args.append("--data.kernel_length=" + str(self.kernel_length))
        args.append("--data.fduration=" + str(self.fduration))
        args.append("--data.fftlength=" + str(fftlength))
        args.append("--data.highpass=" + str(self.highpass))
        args.append("--data.lowpass=" + str(lowpass))
        if self.q is not None:
            args.append("--data.q=" + str(self.q))
        return args

    def get_args(self):
        args = [
            "--config",
            self.train_config,
            "--seed_everything",
            str(self.seed),
            f"--data.ifos=[{','.join(self.ifos)}]",
            "--data.background_dir",
            str(self.background_dir),
            "--data.waveforms_dir",
            str(self.waveforms_dir),
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
        default="ghcr.io/ml4gw/aframe/train:main",
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
