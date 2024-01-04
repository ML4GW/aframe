import law
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.tasks.export.base import ExportParams


@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
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
