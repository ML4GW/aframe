# ruff: noqa: F821
"""Merge per-group inference outputs into the final event sets.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import json
import sys

from infer.aggregate import merge_timeseries
from ledger.events import EventSet, RecoveredInjectionSet

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

# Each group job has already split zero-lag branches out of its background
# when a zero_lag output is requested. Otherwise, everything is in
# background.hdf5 (useful for R&P analysis).
# R&P foregrounds are plain EventSets, not RecoveredInjectionSets.
foreground_cls = (
    EventSet
    if snakemake.params.analysis_type == "rnp"
    else RecoveredInjectionSet
)


def as_list(files):
    """
    Snakemake gives a named input with one file as a string
    """
    return [files] if isinstance(files, str) else list(files)


# Each group's row counts from its metadata, which saves opening every
# group file an extra time to count them.
counts = {}
for fname in as_list(snakemake.input.metadata):
    with open(fname) as f:
        counts[fname] = json.load(f)


def merge(cls, inputs, output, metadata, name):
    """
    Merge the groups' `name` event sets, keeping the group files, which
    are aggregate_infer's declared inputs.
    """
    length = sum(counts[fname][name] for fname in as_list(metadata))
    cls.aggregate(as_list(inputs), output, clean=False, length=length)


# Background (and zero lag) from every group
names = ["background"] + (
    ["zero_lag"] if hasattr(snakemake.output, "zero_lag") else []
)
for name in names:
    merge(
        EventSet,
        snakemake.input[name],
        snakemake.output[name],
        snakemake.input.metadata,
        name,
    )

# Foreground: for R&P, whose frames carry their own injections, one from
# every group; otherwise one per population, from only its groups
if hasattr(snakemake.output, "foreground"):
    merge(
        foreground_cls,
        snakemake.input.foreground,
        snakemake.output.foreground,
        snakemake.input.metadata,
        "foreground",
    )
for population in snakemake.params.populations:
    merge(
        foreground_cls,
        snakemake.input[f"foreground_{population}"],
        snakemake.output[f"foreground_{population}"],
        snakemake.input[f"metadata_{population}"],
        "foreground",
    )

background = EventSet.read(snakemake.output.background)
background = background.sort_by("detection_statistic")
background.write(snakemake.output.background)

# See infer.aggregate.write_timeseries for the file layout.
if hasattr(snakemake.output, "timeseries"):
    merge_timeseries(snakemake.input.timeseries, snakemake.output.timeseries)
