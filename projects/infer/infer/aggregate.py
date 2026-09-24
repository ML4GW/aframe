"""Write and merge network output timeseries.

Each group job merges its branches into a per-group file, and
aggregate_infer merges those into the final output. Event sets are
merged the same way with `Ledger.aggregate`.
"""

from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

TIMESERIES_GROUPS = ("background", "foreground")


def write_timeseries(
    fname: Path,
    branch_id: str,
    background: np.ndarray,
    foreground: np.ndarray | None,
    t0: float,
    sample_t0: float,
    inference_sampling_rate: float,
    shifts: np.ndarray,
) -> None:
    """Write one branch's raw network output timeseries.

    The file has `background` and `foreground` groups holding one dataset
    per branch, keyed by branch id. Each dataset has attributes for t0
    (segment start), sample_t0 (GPS time of the first sample), and shifts.
    The series include the PSD burn-in and are not integrated; see
    `Postprocessor` for the steps that turn them into events.
    An `index` dataset at the root allows easy lookup, and
    inference_sampling_rate is stored as a file-level attribute. The time
    of sample i is sample_t0 + i / inference_sampling_rate.
    """
    if foreground is None:
        foreground = np.zeros(0)
    dtype = np.dtype(
        [
            ("branch_id", np.int32),
            ("t0", np.float64),
            ("sample_t0", np.float64),
            ("shifts", np.float64, (len(shifts),)),
        ]
    )
    index = np.array([(int(branch_id), t0, sample_t0, shifts)], dtype=dtype)
    with h5py.File(fname, "w") as f:
        f.attrs["inference_sampling_rate"] = inference_sampling_rate
        for name, data in zip(
            TIMESERIES_GROUPS, (background, foreground), strict=True
        ):
            ds = f.create_group(name).create_dataset(branch_id, data=data)
            ds.attrs["t0"] = t0
            ds.attrs["sample_t0"] = sample_t0
            ds.attrs["shifts"] = shifts
        f.create_dataset("index", data=index)


def merge_timeseries(files: Iterable[Path], fname: Path) -> None:
    """Merge files written by `write_timeseries` (or by this function)."""
    index = []
    with h5py.File(fname, "w") as out:
        for name in TIMESERIES_GROUPS:
            out.create_group(name)
        for f in files:
            with h5py.File(f, "r") as src:
                out.attrs["inference_sampling_rate"] = src.attrs[
                    "inference_sampling_rate"
                ]
                for name in TIMESERIES_GROUPS:
                    for branch_id in src[name]:
                        src.copy(src[name][branch_id], out[name], branch_id)
                index.append(src["index"][:])
        out.create_dataset("index", data=np.concatenate(index))
