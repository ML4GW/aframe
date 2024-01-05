import law
import luigi
from luigi.util import inherits

from aframe.base import AframeSandbox, AframeSingularityTask


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


class ExportSandbox(AframeSandbox):
    sandbox_type = "aframe_export"

    def _get_volumes(self):
        volumes = super()._get_volumes()
        # hard code cuda version
        volumes["/usr/local/cuda-11.8/bin"] = "/usr/local/cuda-11.8/bin"
        volumes["/usr/local/cuda-11.8/lib64"] = "/usr/local/cuda-11.8/lib64"
        return volumes


@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
    @property
    def sandbox(self):
        return f"aframe_export::{self.image}"

    def output(self):
        # TODO: custom file target that checks for existence
        # of all necessary model repo directories and files
        return law.LocalFileTarget(self.repository_directory)

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
