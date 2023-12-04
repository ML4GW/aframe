import law
import luigi

from aframe.tasks import (
    ExportLocal,
    Fetch,
    GenerateTimeslideWaveforms,
    GenerateWaveforms,
    MergeTimeslideWaveforms,
    TrainLocal,
)
from aframe.tasks.pipelines.sandbox.config import SandboxConfig

config = SandboxConfig()


class SandboxTrainDatagen(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)

    def requires(self):
        yield Fetch.req(
            self, image="data.sif", **config.train_background.to_dict()
        )
        yield GenerateWaveforms.req(
            self, image="data.sif", **config.train_waveforms.to_dict()
        )


class SandboxTrain(TrainLocal):
    def requires(self):
        return SandboxTrainDatagen.req(self)


class SandboxExport(ExportLocal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = self.input().path

    def requires(self):
        # expicitly pass parameters image and config parameters
        # b/c these are common parameters that should
        # not be inherited from the export task
        return SandboxTrain.req(
            self, image="train.sif", **config.train.to_dict()
        )


# requires background training files for calculating snr
class SandboxGenerateTimeslideWaveforms(GenerateTimeslideWaveforms):
    def requires(self):
        return Fetch.req(
            self, image="data.sif", **config.train_background.to_dict()
        )


class SandboxTimeslideWaveforms(MergeTimeslideWaveforms):
    def requires(self):
        return SandboxGenerateTimeslideWaveforms.req(self)


class Sandbox(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    def requires(self):
        yield SandboxExport.req(
            self, image="export.sif", **config.export.to_dict()
        )
        yield SandboxTimeslideWaveforms.req(
            self, image="data.sif", **config.timeslide_waveforms.to_dict()
        )
