# ruff: noqa: F821
"""Merge per-branch inference outputs into the final event sets.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import json
import shutil
import sys
from pathlib import Path

from ledger.events import EventSet, RecoveredInjectionSet

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

with open(snakemake.input.branch_map) as f:
    branch_map = json.load(f)

tmp_dir = Path(snakemake.params.tmp_dir)
background, zero_lag, foreground = [], [], []
background_length, zero_lag_length, foreground_length = 0, 0, 0

# Build list of files to aggregate and compute total lengths for pre-allocation
for branch_id, branch in branch_map.items():
    bg = tmp_dir / branch_id / "background.hdf5"
    fg = tmp_dir / branch_id / "foreground.hdf5"
    with open(tmp_dir / branch_id / "metadata.json") as f:
        meta = json.load(f)
    foreground.append(fg)
    foreground_length += meta["foreground_length"]
    if all(s == 0 for s in branch["shifts"]):
        zero_lag.append(bg)
        zero_lag_length += meta["background_length"]
    else:
        background.append(bg)
        background_length += meta["background_length"]

# Aggregate background, foreground, and potentially zero-lag
EventSet.aggregate(
    background,
    snakemake.output.background,
    clean=True,
    length=background_length,
)
RecoveredInjectionSet.aggregate(
    foreground,
    snakemake.output.foreground,
    clean=True,
    length=foreground_length,
)

if hasattr(snakemake.output, "zero_lag"):
    EventSet.aggregate(
        zero_lag, snakemake.output.zero_lag, clean=True, length=zero_lag_length
    )

background = EventSet.read(snakemake.output.background)
background = background.sort_by("detection_statistic")
background.write(snakemake.output.background)

# If the network timeseries were returned, aggregate them as well.
# We store them in an HDF5 file with groups for background and foreground, and
# datasets for each branch. Each dataset has attributes for t0 (segment start),
# sample_t0 (GPS time of the first sample), and shifts, and there is an index
# dataset at the root of the file for easy lookup. inference_sampling_rate is
# global to the run and stored as a file-level attribute. The time of sample i
# is sample_t0 + i / inference_sampling_rate.
if hasattr(snakemake.output, "timeseries"):
    import h5py
    import numpy as np

    index_rows = []
    with h5py.File(snakemake.output.timeseries, "w") as out:
        bg_grp = out.create_group("background")
        fg_grp = out.create_group("foreground")
        for branch_id in branch_map:
            with h5py.File(tmp_dir / branch_id / "timeseries.hdf5", "r") as g:
                t0 = g.attrs["t0"]
                sample_t0 = g.attrs["sample_t0"]
                sampling_rate = g.attrs["inference_sampling_rate"]
                shifts = g.attrs["shifts"]
                bg_ds = bg_grp.create_dataset(
                    branch_id, data=g["background"][:]
                )
                bg_ds.attrs["t0"] = t0
                bg_ds.attrs["sample_t0"] = sample_t0
                bg_ds.attrs["shifts"] = shifts
                fg_ds = fg_grp.create_dataset(
                    branch_id, data=g["foreground"][:]
                )
                fg_ds.attrs["t0"] = t0
                fg_ds.attrs["sample_t0"] = sample_t0
                fg_ds.attrs["shifts"] = shifts
            index_rows.append((int(branch_id), t0, sample_t0, shifts))
        out.attrs["inference_sampling_rate"] = sampling_rate
        dtype = np.dtype(
            [
                ("branch_id", np.int32),
                ("t0", np.float64),
                ("sample_t0", np.float64),
                ("shifts", np.float64, (len(shifts),)),
            ]
        )
        out.create_dataset("index", data=np.array(index_rows, dtype=dtype))

shutil.rmtree(snakemake.params.tmp_dir, ignore_errors=True)
