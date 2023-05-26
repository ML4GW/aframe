#!/usr/bin/env python3
# coding: utf-8
import os
import shutil
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from aframe.io import h5


@pytest.fixture(scope="function")
def write_dir():
    write_dir = "tmp"
    os.makedirs(write_dir, exist_ok=True)
    yield Path(write_dir)
    shutil.rmtree(write_dir)


@pytest.fixture(scope="function")
def t():
    t = [
        1238163456,
        12345677890,
        12345677890,
        12345677890,
        12345677890,
        1238167552,
    ]
    return t


@pytest.fixture(scope="function")
def t_err():
    t = [
        1238163456,
        12345677890,
        12345677890,
        12345677890,
        12345677890,
        1238167552,
    ]
    return t


@pytest.fixture(scope="function")
def y():
    y = [23, 53.5, 234, 0, 5, 456.345]
    return y


@pytest.fixture(scope="function")
def y_err():
    y_err = [23, 53.5, 234, 0, 5, 456.345, 45.78]
    return y_err


@pytest.fixture(scope="function")
def datasets():
    datasets = {
        "dataset1": np.arange(6),
        "dataset2": np.arange(10, 16),
        "dataset3": np.arange(20, 26),
        "dataset4": np.arange(30, 36),
    }
    return datasets


@pytest.fixture(scope="function")
def datasets_err():
    datasets_err = {
        "dataset1": np.arange(6),
        "dataset2": np.arange(10, 16),
        "dataset3": np.arange(20, 25),
        "dataset4": np.arange(30, 36),
    }
    return datasets_err


@pytest.fixture(scope="function")
def prefix():
    prefix = "out"
    return prefix


def check_file_contents(
    fname, t, prefix, y: Optional["np.ndarray"] = None, **datasets
):
    with h5py.File(fname, "r") as f:
        # check the timestamps array
        assert (t == f["GPSstart"][:]).all()

        # if there is "out" array, it should be checked
        if y:
            assert (y == f[prefix][:]).all()
        for key, value in datasets.items():
            assert key in str(f.keys())
            assert (value == f[key][:]).all()


def check_filename_format(write_dir: "Path", fname: "Path", t, prefix):
    # check the file name format
    length = t[-1] - t[0] + t[1] - t[0]
    assert fname == write_dir / f"{prefix}_{t[0]}-{length}.hdf5"


def test_write_timeseries(write_dir: "Path", t, y, prefix, datasets):

    fname = h5.write_timeseries(write_dir, prefix, t, y, **datasets)

    # check the file name format
    check_filename_format(write_dir, fname, t, prefix)

    # check if file contents were written properly
    check_file_contents(fname, t, prefix, y, **datasets)


# This test checks if the optional parameter y is not provided
# write_timeseries works as desired
def test_write_timeseries_y_Optional(write_dir: "Path", t, prefix, datasets):

    fname = h5.write_timeseries(write_dir, prefix, t, **datasets)
    check_filename_format(write_dir, fname, t, prefix)
    check_file_contents(fname, t, prefix, **datasets)


# This test checks that a value error is thrown if y array
#  is not the same length as t array
def test_write_timeseries_ValueErrorInY(
    write_dir: "Path", t, y_err, prefix, datasets
):

    with pytest.raises(ValueError):
        h5.write_timeseries(write_dir, prefix, t, y_err, **datasets)


# This test checks that a value error is thrown if each dataset array
# is not the same length as t array
def test_write_timeseries_ValueErrorInDatasetLen(
    write_dir: "Path", t, y, prefix, datasets_err
):

    with pytest.raises(ValueError):
        h5.write_timeseries(write_dir, prefix, t, y, **datasets_err)


# This test checks the read_timeseries function
def test_read_timeseries(write_dir: "Path", t, prefix, datasets):

    length = t[-1] - t[0] + t[1] - t[0]
    fname = write_dir / f"{prefix}_{t[0]}-{length}.hdf5"

    # first manually write the timeseries in the required format
    with h5py.File(fname, "w") as f:
        f["GPSstart"] = t
        for key, value in datasets.items():
            f[key] = value

    # test the read_timeseries function
    values = h5.read_timeseries(fname, *datasets)

    # each array in "values", except the last one
    #  should match with dataset arrays
    for key, value in zip(datasets, values[:-1]):
        assert (value == datasets[key]).all()

    # last array in "values" should be the timeseries
    assert (values[-1][:] == t).all()
