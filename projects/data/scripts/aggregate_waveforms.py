# ruff: noqa: F821
"""Merge per-branch waveform files into one file per named output.

Each output is merged from the rule's input of the same name.
Per-branch files are deleted once merged along with their empty directories,
except for responses, which are also used for inference.

Executed via the snakemake `script:` directive.
The `snakemake` object is injected by snakemake.
"""

import sys
from pathlib import Path

from data.cleanup import remove_empty_dirs
from ledger.injections import (
    InjectionParameterSet,
    InterferometerResponseSet,
    WaveformPolarizationSet,
    WaveformSet,
    waveform_class_factory,
)

# `script:` directives are not auto-redirected to the rule's log, so
# send stdout/stderr there to capture tracebacks on failure.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

ifos = snakemake.params.ifos
CLASSES = {
    "responses": waveform_class_factory(
        ifos, InterferometerResponseSet, "ResponseSet"
    ),
    "waveforms": waveform_class_factory(ifos, WaveformSet, "WaveformSet"),
    "polarizations": WaveformPolarizationSet,
    "parameters": InjectionParameterSet,
}

for name, output in snakemake.output.items():
    kind = snakemake.params.classes[name]
    cls = CLASSES[kind]
    files = snakemake.input[name]
    # snakemake gives a named input with one file as a string
    if isinstance(files, str):
        files = [files]
    files = [Path(f) for f in files]
    # Don't clean up responses because inference reads the individual branches
    cls.aggregate(files, output, clean=kind != "responses")
    remove_empty_dirs(files, Path(output).parent)
