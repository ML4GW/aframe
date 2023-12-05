import os

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
    dev = luigi.BoolParameter(default=False, significant=False)

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


class SandboxGenerateTimeslideWaveforms(GenerateTimeslideWaveforms):
    def workflow_requires(self):
        reqs = {}
        # requires a background training segment for calculating snr
        reqs["train"] = Fetch.req(
            self, image="data.sif", **config.train_background.to_dict()
        )
        # requires background testing segments
        # to determine number of waveforms to generate
        reqs["test"] = Fetch.req(
            self, image="data.sif", **config.test_background.to_dict()
        )
        return reqs


class SandboxTimeslideWaveforms(MergeTimeslideWaveforms):
    @property
    def condor_directory(self):
        data_dir = config.timeslide_waveforms.data_dir
        return os.path.join(data_dir, "test", "condor")

    def requires(self):
        return SandboxGenerateTimeslideWaveforms.req(
            self,
            image="data.sif",
            condor_directory=self.condor_directory,
            **config.timeslide_waveforms.to_dict()
        )


class Sandbox(law.WrapperTask):
    dev = luigi.BoolParameter(default=False, significant=False)
    gpus = luigi.Parameter(default="", significant=False)

    def requires(self):
        yield SandboxExport.req(
            self, image="export.sif", **config.export.to_dict()
        )

        yield SandboxTimeslideWaveforms.req(
            self, image="data.sif", **config.timeslide_waveforms.to_dict()
        )
