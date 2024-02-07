import os

import law
import luigi
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.tasks.export.target import ModelRepositoryTarget


class ExportParams(law.Task):
    weights = luigi.Parameter(default="")
    fduration = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    repository_directory = luigi.Parameter()
    streams_per_gpu = luigi.IntParameter()
    aframe_instances = luigi.IntParameter()
    clean = luigi.BoolParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    fftlength = luigi.FloatParameter(default=0)
    ifos = luigi.ListParameter(default=["H1", "L1"])
    # TODO: resolve enum platform parsing error
    # platform = luigi.Parameter(default="TENSORRT")


@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.repository_directory, exist_ok=True)

    def output(self):
        return ModelRepositoryTarget(self.repository_directory)

    def requires(self):
        raise NotImplementedError

    @property
    def default_image(self):
        return "export.sif"

    @property
    def num_ifos(self):
        return len(self.ifos)

    def run(self):
        from export.main import export

        if not self.fftlength:
            self.fftlength = None

        weights = self.input().path

        export(
            weights,
            self.repository_directory,
            self.num_ifos,
            self.kernel_length,
            self.inference_sampling_rate,
            self.sample_rate,
            self.batch_size,
            self.fduration,
            self.psd_length,
            self.fftlength,
            self.highpass,
            self.streams_per_gpu,
            self.aframe_instances,
            # self.platform,
            clean=self.clean,
            # verbose=self.verbose,
        )
