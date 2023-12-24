import law

from aframe.base import AframeSingularityTask
from aframe.tasks.export.base import ExportParams


class ExportLocal(AframeSingularityTask, ExportParams):
    def output(self):
        # TODO: custom file target that checks for existence
        # of all necessary model repo directories and files
        return law.LocalFileTarget(self.repository_directory)

    def run(self):
        from export.main import export

        with self.input().open("rb") as f:
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
                self.platform,
                self.clean,
                self.verbose,
            )
