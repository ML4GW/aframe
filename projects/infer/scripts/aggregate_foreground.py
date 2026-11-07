# ruff: noqa: F821
"""Merge groups' foregrounds into one foreground: a population's groups', or
for R&P every group's.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import json
import sys

from ledger.events import EventSet, RecoveredInjectionSet

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

# R&P foregrounds are plain EventSets, not RecoveredInjectionSets.
foreground_cls = (
    EventSet
    if snakemake.params.analysis_type == "rnp"
    else RecoveredInjectionSet
)


def as_list(files):
    """snakemake gives a named input with one file as a string"""
    return [files] if isinstance(files, str) else list(files)


# The groups' row counts from their metadata, which saves opening every
# group file an extra time to count them.
length = 0
for fname in as_list(snakemake.input.metadata):
    with open(fname) as f:
        length += json.load(f)["foreground"]

# Keep the group files, which are aggregate_foreground's declared inputs.
foreground_cls.aggregate(
    as_list(snakemake.input.foreground),
    snakemake.output.foreground,
    clean=False,
    length=length,
)
