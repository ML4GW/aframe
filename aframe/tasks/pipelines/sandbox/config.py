import os

import law
import luigi
from luigi import Config as _Config

from aframe.config import Defaults

project_base = "/opt/aframe/projects"


class Config(_Config):
    def to_dict(self):
        return {key: getattr(self, key) for key in self.get_param_names()}


# BIG TODO: law supports
# 1. parsing environment variables
# 2. referencing other config sections (i.e. a base)
# so, we can remove the "hack" where we infer
# base parameters via default values


# base config that stores parameters
# common to multiple tasks
class base(luigi.Config):
    # general parameters
    ifos = luigi.ListParameter(default=["H1", "L1"])
    run_dir = luigi.Parameter(default=os.getenv("AFRAME_RUN_DIR", ""))
    data_dir = law.Parameter(default=os.getenv("AFRAME_DATA_DIR", ""))
    # data generation parameters
    train_start = luigi.FloatParameter()
    train_stop = luigi.FloatParameter()
    test_stop = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter(default=2048)
    # data conditioning / preprocessing parameters
    fduration = luigi.FloatParameter(default=2)
    fftlength = luigi.FloatParameter(default=2.0)
    highpass = luigi.FloatParameter(default=32.0)
    # training parameters
    kernel_length = luigi.FloatParameter()
    # waveform parameters
    waveform_approximant = luigi.Parameter(default="IMRPhenomPv2")
    waveform_duration = luigi.FloatParameter(default=8.0)
    minimum_frequency = luigi.FloatParameter(default=20)
    reference_frequency = luigi.FloatParameter(default=20)
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


class train_background(Config):
    start = luigi.FloatParameter(default=base().train_start)
    end = luigi.FloatParameter(default=base().train_stop)
    segments_file = luigi.Parameter(
        default=os.path.join(base().data_dir, "train", "segments.txt")
    )
    data_dir = luigi.Parameter(
        os.path.join(base().data_dir, "train", "background")
    )
    condor_directory = luigi.Parameter(
        os.path.join(base().data_dir, "train", "condor")
    )
    sample_rate = luigi.FloatParameter(default=base().sample_rate)
    min_duration = luigi.FloatParameter()
    max_duration = luigi.FloatParameter(default=-1)
    flags = luigi.ListParameter()
    channels = luigi.ListParameter()


class test_background(Config):
    start = luigi.FloatParameter(default=base().train_stop)
    end = luigi.FloatParameter(default=base().test_stop)
    segments_file = luigi.Parameter(
        default=os.path.join(base().data_dir, "test", "segments.txt")
    )
    data_dir = luigi.Parameter(
        os.path.join(base().data_dir, "test", "background")
    )
    condor_directory = luigi.Parameter(
        os.path.join(base().data_dir, "test", "condor")
    )
    sample_rate = luigi.FloatParameter(default=base().sample_rate)
    min_duration = luigi.FloatParameter()
    max_duration = luigi.FloatParameter(default=-1)
    flags = luigi.ListParameter()
    channels = luigi.ListParameter()


class train_waveforms(Config):
    num_signals = luigi.IntParameter()
    waveform_duration = luigi.FloatParameter()
    prior = luigi.Parameter()
    sample_rate = luigi.FloatParameter(default=base().sample_rate)
    output_file = luigi.Parameter(
        default=os.path.join(base().data_dir, "train", "signals.hdf5")
    )
    minimum_frequency = luigi.FloatParameter(default=base().minimum_frequency)
    reference_frequency = luigi.FloatParameter(
        default=base().reference_frequency
    )
    waveform_approximant = luigi.Parameter(default=base().waveform_approximant)


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


class timeslide_waveforms(Config):
    shifts = luigi.ListParameter()
    spacing = luigi.FloatParameter()
    buffer = luigi.FloatParameter()
    prior = luigi.Parameter()
    snr_threshold = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    start = luigi.FloatParameter(default=base().train_stop)
    end = luigi.FloatParameter(default=base().test_stop)
    ifos = luigi.ListParameter(default=base().ifos)
    psd_length = luigi.FloatParameter(default=base().inference_psd_length)
    sample_rate = luigi.FloatParameter(default=base().sample_rate)
    minimum_frequency = luigi.FloatParameter(default=base().minimum_frequency)
    reference_frequency = luigi.FloatParameter(
        default=base().reference_frequency
    )
    waveform_duration = luigi.FloatParameter(default=base().waveform_duration)
    waveform_approximant = luigi.Parameter(default=base().waveform_approximant)
    highpass = luigi.FloatParameter(default=base().highpass)
    data_dir = luigi.Parameter(
        default=os.path.join(base().data_dir, "test", "timeslide_waveforms")
    )
    output_dir = luigi.Parameter(default=os.path.join(base().data_dir, "test"))
    seed = luigi.IntParameter(default=101588)


class SandboxConfig(luigi.Config):
    train_background = train_background()
    train_waveforms = train_waveforms()
    export = export()
    train = train()
    timeslide_waveforms = timeslide_waveforms()
    test_background = test_background()
