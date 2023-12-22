import os

import law
import luigi

from aframe.pipelines.sandbox.config import SandboxConfig
from aframe.tasks import (
    ExportLocal,
    Fetch,
    GenerateTimeslideWaveforms,
    GenerateWaveforms,
    InferLocal,
    MergeTimeslideWaveforms,
    TrainLocal,
)

config = SandboxConfig()


class SandboxTrainDatagen(law.WrapperTask):
    dev = luigi.BoolParameter(default=False, significant=False)

    def requires(self):
        yield Fetch.req(
            self,
            image="data.sif",
            condor_directory=os.path.join(config.data_dir, "condor", "train"),
            data_dir=os.path.join(config.data_dir, "train", "background"),
            segments_file=os.path.join(
                config.data_dir, "train", "segments.txt"
            ),
            **config.train_background.to_dict(),
        )
        yield GenerateWaveforms.req(
            self,
            image="data.sif",
            output_file=os.path.join(config.data_dir, "train", "signals.hdf5"),
            **config.train_waveforms.to_dict(),
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
            self,
            image="train.sif",
            data_dir=os.path.join(config.data_dir, "train"),
            run_dir=config.run_dir,
            **config.train.to_dict(),
        )


class SandboxGenerateTimeslideWaveforms(GenerateTimeslideWaveforms):
    def workflow_requires(self):
        reqs = {}
        # requires background testing segments
        # to determine number of waveforms to generate
        reqs["test_segments"] = Fetch.req(
            self,
            image="data.sif",
            condor_directory=os.path.join(config.data_dir, "condor", "test"),
            data_dir=os.path.join(config.data_dir, "test", "background"),
            segments_file=os.path.join(
                config.data_dir, "test", "segments.txt"
            ),
            **config.test_background.to_dict(),
        )
        return reqs

    def requires(self):
        reqs = {}
        # requires a background training segment for calculating snr
        # TODO: how to specify just the last segment?
        reqs["train_segments"] = Fetch.req(
            self,
            branch=-1,
            image="data.sif",
            condor_directory=os.path.join(config.data_dir, "condor", "train"),
            data_dir=os.path.join(config.data_dir, "train", "background"),
            segments_file=os.path.join(
                config.data_dir, "train", "segments.txt"
            ),
            **config.train_background.to_dict(),
        )
        return reqs


class SandboxTimeslideWaveforms(MergeTimeslideWaveforms):
    @property
    def data_dir(self):
        return os.path.join(config.data_dir, "test")

    @property
    def output_dir(self):
        return os.path.join(config.data_dir, "timeslide_waveforms")

    @property
    def condor_directory(self):
        return os.path.join(config.data_dir, "condor", "timeslide_waveforms")

    def requires(self):
        return SandboxGenerateTimeslideWaveforms.req(
            self,
            image="data.sif",
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            condor_directory=self.condor_directory,
            **config.timeslide_waveforms.to_dict(),
        )


class SandboxInfer(InferLocal):
    def requires(self):
        reqs = {}
        reqs["export"] = SandboxExport.req(
            self,
            image="export.sif",
            repository_directory=os.path.join(
                config.base.run_dir, "model_repo"
            ),
            logfile=os.path.join(config.base.log_dir, "export.log"),
            **config.export.to_dict(),
        )
        reqs["waveforms"] = SandboxTimeslideWaveforms.req(
            self, image="data.sif", **config.timeslide_waveforms.to_dict()
        )
        reqs["data"] = Fetch.req(
            self,
            image="data.sif",
            condor_directory=os.path.join(config.data_dir, "condor", "test"),
            data_dir=os.path.join(config.data_dir, "test", "background"),
            segments_file=os.path.join(
                config.data_dir, "test", "segments.txt"
            ),
            **config.test_background.to_dict(),
        )
        return reqs


class Sandbox(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    def requires(self):
        # simply call infer, which will
        # call all necessary downstream tasks!
        yield SandboxInfer.req(
            self,
            output_dir=config.infer.output_dir,
            **config.infer.to_dict(),
        )
