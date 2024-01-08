import law
import luigi
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.targets import s3_or_local


class ExportParams(law.Task):
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
    ifos = luigi.ListParameter(default=["H1", "L1"])
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    logfile = luigi.Parameter()


@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
    def output(self):
        # TODO: custom file target that checks for existence
        # of all necessary model repo directories and files
        return law.LocalFileTarget(self.repository_directory)

    def input(self):
        return s3_or_local(self.weights, client=None)

    @property
    def num_ifos(self):
        return len(self.ifos)

    def run(self):
        from export.main import export

        with self.input().open("r") as f:
            export(
                f,
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
