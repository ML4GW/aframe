"""Top-level Snakefile

Usage, from the root directory:
    snakemake -n                                   # dry-run
    snakemake --profile pipeline/profiles/condor   # HTCondor
    snakemake --profile pipeline/profiles/local    # local execution (dev)

The pipeline config is loaded from pipeline/config/config.yaml by
default. To use a different config, copy the original, make
modifications, and run:

    snakemake --configfile my_run.yaml --profile pipeline/profiles/condor
"""

from pathlib import Path


configfile: "pipeline/config/config.yaml"


if "run_dir" not in config:
    raise WorkflowError(
        "'run_dir' must be set in your config. "
        "Pass it via --configfile or --config run_dir=..."
    )

config.setdefault("background_dir", str(Path(config["run_dir"]) / "data"))
config.setdefault("waveforms_dir", str(Path(config["run_dir"]) / "waveforms"))
config.setdefault("log_dir", str(Path(config["run_dir"]) / "logs"))

run_dir = Path(config["run_dir"])
log_dir = Path(config["log_dir"])


include: "projects/data/data.smk"
include: "projects/train/train.smk"
include: "projects/export/export.smk"
include: "projects/infer/infer.smk"
include: "projects/plots/plots.smk"


rule all:
    default_target: True
    input:
        rules.stop_triton.output,
        rules.sensitive_volume.output,
