import law
import luigi

from aframe.config import Defaults


class ExportParams(law.Task):
    config = luigi.Parameter(default=Defaults.EXPORT)
    weights = luigi.Parameter(default="")
    fftlength = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    repository_directory = luigi.Parameter()
    streams_per_gpu = luigi.IntParameter()
    aframe_instances = luigi.IntParameter()
    # TODO: resolve enum platform parsing error
    # platform = luigi.Parameter(default="TENSORRT")
    clean = luigi.BoolParameter()
    ifos = luigi.ListParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    logfile = luigi.Parameter()
