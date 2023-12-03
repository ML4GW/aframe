import os

import luigi
from luigi import Config as _Config

from aframe.config import Defaults

project_base = "/opt/aframe/projects"


class Config(_Config):
    def to_dict(self):
        return {key: getattr(self, key) for key in self.get_param_names()}


# base config that stores parameters
# common to multiple tasks
class base(Config):
    # general parameters
    ifos = luigi.ListParameter(default=["H1", "L1"])
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    # data generation parameters
    sample_rate = luigi.FloatParameter(default=2048)
    # data conditioning / preprocessing parameters
    fduration = luigi.FloatParameter(default=2)
    fftlength = luigi.FloatParameter(default=2.0)
    highpass = luigi.FloatParameter(default=32.0)
    # training parameters
    kernel_length = luigi.FloatParameter()
    # waveform parameters
    waveform_duration = luigi.FloatParameter(default=8.0)
    # inference / export parameters
    inference_psd_length = luigi.FloatParameter(default=64)
    inference_sampling_rate = luigi.FloatParameter(default=16)
    inference_batch_size = luigi.IntParameter(default=512)

    @property
    def log_dir(self):
        return os.path.join(self.run_dir, "logs")

    @property
    def num_ifos(self):
        return len(self.ifos)


class train(Config):
    data_dir = luigi.Parameter(os.path.join(base().data_dir, "train"))
    run_dir = luigi.Parameter(os.path.join(base().run_dir, "train"))
    config = luigi.Parameter(default=Defaults.TRAIN)
    ifos = luigi.ListParameter(default=base().ifos)
    kernel_length = luigi.FloatParameter(default=base().kernel_length)
    highpass = luigi.FloatParameter(default=base().highpass)
    fduration = luigi.FloatParameter(default=base().fduration)
    seed = luigi.IntParameter(default=101588)
    use_wandb = luigi.BoolParameter(default=False)


class export(Config):
    fftlength = luigi.FloatParameter(default=base().fftlength)
    fduration = luigi.FloatParameter(default=base().fduration)
    kernel_length = luigi.FloatParameter(default=base().kernel_length)
    inference_sampling_rate = luigi.FloatParameter(
        default=base().inference_sampling_rate
    )
    sample_rate = luigi.FloatParameter(default=base().sample_rate)
    fduration = luigi.FloatParameter(default=base().fduration)
    repository_directory = luigi.Parameter(
        default=os.path.join(base().run_dir, "model_repo")
    )

    # TODO: resolve enum platform parsing error
    # platform = luigi.Parameter(default="TENSORRT")
    num_ifos = luigi.IntParameter(default=base().num_ifos)
    batch_size = luigi.IntParameter(default=base().inference_batch_size)
    psd_length = luigi.FloatParameter(default=base().inference_psd_length)
    highpass = luigi.FloatParameter(default=base().highpass)
    logfile = luigi.Parameter(
        default=os.path.join(base().log_dir, "export.log")
    )

    clean = luigi.BoolParameter(default=False)
    streams_per_gpu = luigi.IntParameter(default=2)
    aframe_instances = luigi.IntParameter(default=2)


class SandboxConfig(luigi.Config):
    export = export()
    train = train()
