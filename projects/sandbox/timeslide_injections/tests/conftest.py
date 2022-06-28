import logging
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def outdir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)
