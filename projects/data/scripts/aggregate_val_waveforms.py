# ruff: noqa: F821
"""Merge per-branch validation waveform files into val_waveforms.hdf5.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import shutil
import sys
from pathlib import Path

from ledger.injections import WaveformSet, waveform_class_factory

# `script:` directives are not auto-redirected to the rule's log, so
# send stdout/stderr there to capture tracebacks on failure.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

cls = waveform_class_factory(
    snakemake.params.ifos,
    WaveformSet,
    "WaveformSet",
)

cls.aggregate(
    [Path(i) for i in list(snakemake.input)],
    snakemake.output[0],
    clean=True,
)

shutil.rmtree(snakemake.params.tmp_dir, ignore_errors=True)
