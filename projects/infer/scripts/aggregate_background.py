# ruff: noqa: F821
"""Merge every group's background into the final background, and its zero
lag and timeseries when they're requested.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import json
import sys

from infer.aggregate import merge_timeseries
from ledger.events import EventSet

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

# Each group job has already split zero-lag branches out of its background
# when a zero_lag output is requested. Otherwise, everything is in
# background.hdf5 (useful for R&P analysis).


# Total row counts from each group's metadata, which saves opening
# every group file an extra time to count them.
lengths = {}
for fname in snakemake.input.metadata:
    with open(fname) as f:
        for name, length in json.load(f).items():
            lengths[name] = lengths.get(name, 0) + length

# Background (and zero lag) from every group
names = ["background"]
if hasattr(snakemake.output, "zero_lag"):
    names.append("zero_lag")
# Keep the group files, which are aggregate_background's declared inputs.
for name in names:
    EventSet.aggregate(
        snakemake.input[name],
        snakemake.output[name],
        clean=False,
        length=lengths[name],
    )

background = EventSet.read(snakemake.output.background)
background = background.sort_by("detection_statistic")
background.write(snakemake.output.background)

# See infer.aggregate.write_timeseries for the file layout.
if hasattr(snakemake.output, "timeseries"):
    merge_timeseries(snakemake.input.timeseries, snakemake.output.timeseries)
