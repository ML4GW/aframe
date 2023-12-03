import law

from aframe.base import AframeBaseParams
from aframe.tasks import ExportLocal, TrainLocal
from aframe.tasks.pipelines.sandbox.config import SandboxConfig

config = SandboxConfig()


# inherit all parameters from the TrainLocal task
class SandboxExport(ExportLocal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = self.input().path

    def requires(self):
        # expicitly pass parameters image and config parameters
        # b/c these are common parameters that should
        # not be inherited from the export task
        return TrainLocal.req(
            self, image="train.sif", **config.train.to_dict()
        )


class Sandbox(law.WrapperTask, AframeBaseParams):
    def requires(self):
        yield SandboxExport.req(
            self, image="export.sif", **config.export.to_dict()
        )
