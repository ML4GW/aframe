import os

import law
import luigi

from aframe.tasks.data import GenerateWaveforms
from aframe.tasks.data.fetch import FetchTrain

"""
Common combinations of tasks or configs
several differents pipelines may find useful
"""


class paths(luigi.Config):
    train_datadir = luigi.Parameter(default=os.getenv("AFRAME_TRAIN_DATA_DIR"))
    test_datadir = luigi.Parameter(default=os.getenv("AFRAME_TEST_DATA_DIR"))
    rundir = luigi.Parameter(default=os.getenv("AFRAME_RUN_DIR"))
    condordir = luigi.Parameter(default=os.getenv("AFRAME_CONDOR_DIR"))


class TrainWaveforms(GenerateWaveforms):
    pass


class TrainDatagen(law.WrapperTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    def requires(self):
        yield FetchTrain.req(
            self,
            segments_file=os.path.join(
                paths().condordir, "train", "segments.txt"
            ),
            data_dir=os.path.join(paths().train_datadir, "background"),
            condor_directory=os.path.join(paths().condordir, "train"),
        )
        yield TrainWaveforms.req(
            self,
            output_file=os.path.join(paths().train_datadir, "signals.hdf5"),
        )
