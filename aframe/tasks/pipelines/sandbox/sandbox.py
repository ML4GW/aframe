import law

from aframe.base import AframeBaseParams
from aframe.tasks import ExportLocal, Fetch, GenerateWaveforms, TrainLocal
from aframe.tasks.pipelines.sandbox.config import SandboxConfig

config = SandboxConfig()


class SandboxTrainDatagen:
    def requires(self):
        yield Fetch.req(
            self, image="fetch.sif", **config.train_background.to_dict()
        )
        yield GenerateWaveforms.req(
            self, image="generate.sif", **config.train_waveforms.to_dict()
        )


class SandboxTrain(TrainLocal):
    def requires(self):
        return SandboxTrainDatagen.req(
            self, image="data.sif", **config.train_background.to_dict()
        )


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


class Sandbox(law.WrapperTask, AframeBaseParams):
    def requires(self):
        yield SandboxExport.req(
            self, image="export.sif", **config.export.to_dict()
        )
