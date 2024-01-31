import os

import law
import luigi

from aframe.tasks import (
    ExportLocal,
    InferLocal,
    TimeslideWaveforms,
    TrainLocal,
)
from aframe.tasks.data import GenerateWaveforms
from aframe.tasks.data.fetch import FetchTrain


class TrainWaveforms(GenerateWaveforms):
    pass


class paths(luigi.Config):
    train_datadir = luigi.Parameter(default=os.getenv("AFRAME_TRAIN_DATA_DIR"))
    test_datadir = luigi.Parameter(default=os.getenv("AFRAME_TEST_DATA_DIR"))
    rundir = luigi.Parameter(default=os.getenv("AFRAME_RUN_DIR"))
    condordir = luigi.Parameter(default=os.getenv("AFRAME_CONDOR_DIR"))


class TrainDatagen(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    def requires(self):
        yield FetchTrain.req(
            self,
            segments_file=os.path.join(paths().train_datadir, "segments.txt"),
            data_dir=os.path.join(paths().train_datadir, "background"),
            condor_directory=os.path.join(paths().condordir, "train"),
        )
        yield TrainWaveforms.req(
            self,
            output_file=os.path.join(paths().train_datadir, "signals.hdf5"),
        )


class Train(TrainLocal):
    def requires(self):
        return TrainDatagen.req(self)


class SandboxExport(ExportLocal):
    def requires(self):
        return Train.req(
            self,
            image="train.sif",
            data_dir=paths().train_datadir,
            run_dir=os.path.join(paths().rundir, "train"),
        )


class SandboxInfer(InferLocal):
    def requires(self):
        reqs = {}
        reqs["model_repository"] = SandboxExport.req(
            self,
            repository_directory=os.path.join(paths().rundir, "model_repo"),
        )
        ts_waveforms = TimeslideWaveforms.req(
            self,
            output_dir=paths().test_datadir,
            condor_directory=os.path.join(paths().condordir),
        )
        fetch = ts_waveforms.requires().workflow_requires()["test_segments"]

        reqs["data"] = fetch
        reqs["waveforms"] = ts_waveforms
        return reqs


class Sandbox(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    def requires(self):
        # simply call infer, which will
        # call all necessary downstream tasks!
        yield SandboxInfer.req(
            self,
            output_dir=os.path.join(paths().rundir, "infer"),
        )
