import os

import luigi

from aframe.parameters import PathParameter

"""
Common combinations of tasks or configs
several differents pipelines may find useful
"""


class paths(luigi.Config):
    train_datadir = PathParameter(default=os.getenv("AFRAME_TRAIN_DATA_DIR"))
    train_rundir = PathParameter(default=os.getenv("AFRAME_TRAIN_RUN_DIR"))
    results_dir = PathParameter(default=os.getenv("AFRAME_RESULTS_DIR"))
    test_datadir = PathParameter(default=os.getenv("AFRAME_TEST_DATA_DIR"))
    condordir = PathParameter(default=os.getenv("AFRAME_CONDOR_DIR"))
