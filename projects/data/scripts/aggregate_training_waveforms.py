# ruff: noqa: F821
"""Merge per-branch training waveform files into training_waveforms.hdf5.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import shutil
import sys
from pathlib import Path

from ledger.injections import WaveformPolarizationSet

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

WaveformPolarizationSet.aggregate(
    [Path(i) for i in list(snakemake.input)],
    snakemake.output[0],
    clean=True,
)

shutil.rmtree(snakemake.params.tmp_dir, ignore_errors=True)
