import logging

import pytest


@pytest.fixture(scope="function")
def datadir(tmp_path):
    datadir = tmp_path / "data"
    datadir.mkdir(parents=True, exist_ok=False)
    return datadir


@pytest.fixture(scope="function")
def logdir(tmp_path):
    logdir = tmp_path / "log"
    logdir.mkdir(parents=True, exist_ok=False)
    yield logdir
    logging.shutdown()
