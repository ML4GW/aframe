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


class base(luigi.Config):
    # general parameters
    seed = luigi.IntParameter(default=1122)
    ifos = luigi.ListParameter()
    run_dir = luigi.Parameter()
    data_dir = luigi.Parameter()
    prior = luigi.Parameter()
    # data generation parameters
    train_start = luigi.FloatParameter()
    train_stop = luigi.FloatParameter()
    test_stop = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    min_duration = luigi.FloatParameter()
    max_duration = luigi.FloatParameter()
    flags = luigi.ListParameter()
    channels = luigi.ListParameter()
    # data conditioning / preprocessing parameters
    fduration = luigi.FloatParameter()
    fftlength = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    # training parameters
    batch_size = luigi.IntParameter()
    kernel_length = luigi.FloatParameter()
    # waveform parameters
    waveform_approximant = luigi.Parameter()
    waveform_duration = luigi.FloatParameter()
    minimum_frequency = luigi.FloatParameter()
    reference_frequency = luigi.FloatParameter()
    # inference / export parameters
    inference_psd_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    inference_batch_size = luigi.IntParameter()

    @property
    def log_dir(self):
        return os.path.join(self.run_dir, "logs")

    @property
    def num_ifos(self):
        return len(self.ifos)


class train_background(Config):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    min_duration = luigi.FloatParameter()
    max_duration = luigi.FloatParameter()
    flags = luigi.ListParameter()
    channels = luigi.ListParameter()


class test_background(train_background):
    pass


class train_waveforms(Config):
    num_signals = luigi.IntParameter()
    waveform_duration = luigi.FloatParameter()
    prior = luigi.Parameter()
    sample_rate = luigi.FloatParameter()
    minimum_frequency = luigi.FloatParameter()
    reference_frequency = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()


class train(Config):
    ifos = luigi.ListParameter()
    kernel_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    use_wandb = luigi.BoolParameter(default=False)
    config = luigi.Parameter(default=Defaults.TRAIN)
    seed = luigi.IntParameter(base().seed)


class export(Config):
    fftlength = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    # TODO: resolve enum platform parsing error
    # platform = luigi.Parameter(default="TENSORRT")
    ifos = luigi.ListParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    clean = luigi.BoolParameter()
    streams_per_gpu = luigi.IntParameter()
    aframe_instances = luigi.IntParameter()


class timeslide_waveforms(Config):
    shifts = luigi.ListParameter()
    spacing = luigi.FloatParameter()
    buffer = luigi.FloatParameter()
    prior = luigi.Parameter()
    snr_threshold = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    ifos = luigi.ListParameter()
    psd_length = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    minimum_frequency = luigi.FloatParameter()
    reference_frequency = luigi.FloatParameter()
    waveform_duration = luigi.FloatParameter()
    waveform_approximant = luigi.Parameter()
    highpass = luigi.FloatParameter()
    seed = luigi.IntParameter(base().seed)


class SandboxConfig(luigi.Config):
    base = base()
    train_background = train_background()
    train_waveforms = train_waveforms()
    export = export()
    train = train()
    timeslide_waveforms = timeslide_waveforms()
    test_background = test_background()

    @property
    def data_dir(self):
        return self.base.data_dir

    @property
    def run_dir(self):
        return self.base.run_dir
