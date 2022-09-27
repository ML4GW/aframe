import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def data_length():
    return 128


@pytest.fixture
def offset():
    # TODO: explore more values
    return 0


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    return request.param


@pytest.fixture(scope="function")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param
