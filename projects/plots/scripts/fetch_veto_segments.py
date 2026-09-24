# ruff: noqa: F821
"""Query the veto segments for the analyzed span into the run directory.

Executed via the snakemake `script:` directive, as a localrule because the
segment database is not reachable from every execute point. The output
is the segments cache that `sensitive_volume` reads instead of querying.
"""

import sys
from pathlib import Path

from ledger.events import EventSet
from plots.main import analysis_span
from plots.vetos.masks import load_or_fetch_segments

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

background = EventSet.read(snakemake.input.background)
start, stop = analysis_span(background)
load_or_fetch_segments(
    snakemake.params.vetos,
    snakemake.params.ifos,
    start,
    stop,
    cache=Path(snakemake.output[0]),
)
