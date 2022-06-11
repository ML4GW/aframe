from typing import TYPE_CHECKING, Optional, Tuple

import h5py

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def write_timeseries(
    write_dir: "Path",
    prefix: str = "out",
    t: "np.ndarray" = None,
    y: Optional["np.ndarray"] = None,
    **datasets,
):
    # read time information from time array, and map
    # to ints for string formatting purposes if the
    # times actually amount to ints
    t0 = t[0]
    t0 = int(t0) if int(t0) == t0 else t0

    length = t[-1] - t[0] + t[1] - t[0]
    length = int(length) if int(length) == length else length

    # format the filename and write the data to an archive
    fname = write_dir / f"{prefix}_{t0}-{length}.hdf5"

    # check the lengths of all data arrays to see
    # if they match the length of timeseries array

    with h5py.File(fname, "w") as f:
        f["GPSstart"] = t
        if y is not None:
            if len(y) != len(t):
                raise ValueError("Length of y and t doesn't match")
            f["out"] = y

        for key, value in datasets.items():
            if len(value) != len(t):
                raise ValueError(
                    f"Length of data array '{key}' "
                    f"doesn't match the length of timeseries",
                )
            f[key] = value

    return fname


def read_timeseries(fname: str, *datasets) -> Tuple["np.ndarray", ...]:
    with h5py.File(fname, "r") as f:
        t = f["GPSstart"][:]

        outputs = []
        for dataset in datasets:
            outputs.append(f[dataset][:].reshape(-1))
    return tuple(outputs) + (t,)
