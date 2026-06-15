# ruff: noqa: F821
"""Merge per-branch testing waveform files into the final injection set.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import shutil
import sys
from pathlib import Path

from ledger.injections import (
    InjectionParameterSet,
    InterferometerResponseSet,
    waveform_class_factory,
)

# `script:` directives are not auto-redirected to the rule's log, so
# send stdout/stderr there to capture tracebacks on failure.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

cls = waveform_class_factory(
    snakemake.params.ifos,
    InterferometerResponseSet,
    "ResponseSet",
)

cls.aggregate(
    [Path(i) for i in snakemake.input.waveforms],
    snakemake.output.waveforms,
    clean=True,
)
InjectionParameterSet.aggregate(
    [Path(i) for i in snakemake.input.rejected],
    snakemake.output.rejected,
    clean=True,
)

shutil.rmtree(snakemake.params.tmp_dir, ignore_errors=True)
