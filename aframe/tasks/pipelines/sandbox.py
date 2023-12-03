import os

import law
import luigi
from luigi.util import inherits

from aframe.config import Defaults
from aframe.config import aframe as Config
from aframe.tasks import ExportLocal, TrainLocal


# inherit all parameters from the TrainLocal task
@inherits(TrainLocal)
class SandboxExport(ExportLocal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = self.input().path

    def requires(self):
        # expicitly pass parameters image and config parameters
        # b/c these are common parameters that should
        # not be inherited from the export task
        return TrainLocal.req(self, image="train.sif", config=Defaults.TRAIN)


class Sandbox(law.WrapperTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")
    cfg = Config()

    def requires(self):
        yield SandboxExport.req(self, image="export.sif")
