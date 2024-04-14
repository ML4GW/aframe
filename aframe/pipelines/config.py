import os

import luigi

"""
Common combinations of tasks or configs
several differents pipelines may find useful
"""


class paths(luigi.Config):
    train_datadir = luigi.Parameter(default=os.getenv("AFRAME_TRAIN_DATA_DIR"))
    train_rundir = luigi.Parameter(default=os.getenv("AFRAME_TRAIN_RUN_DIR"))
    results_dir = luigi.Parameter(default=os.getenv("AFRAME_RESULTS_DIR"))
    test_datadir = luigi.Parameter(default=os.getenv("AFRAME_TEST_DATA_DIR"))
    condordir = luigi.Parameter(default=os.getenv("AFRAME_CONDOR_DIR"))
