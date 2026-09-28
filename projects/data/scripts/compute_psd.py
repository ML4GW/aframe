# ruff: noqa: F821
"""Compute the PSDs that waveform jobs rejection-sample against.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import sys

from data.waveforms.utils import compute_psds, write_psds

# `script:` directives are not auto-redirected to the rule's log, so
# send stdout/stderr there to capture tracebacks on failure.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

ifos, df = snakemake.params.ifos, snakemake.params.df
psds = compute_psds(snakemake.input[0], ifos, df)
write_psds(psds, snakemake.output[0], ifos, df)
