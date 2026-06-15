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
config.setdefault("waveforms_dir", str(Path(config["background_dir"]) / "waveforms"))
config.setdefault("log_dir", str(Path(config["run_dir"]) / "logs"))


include: "projects/data/data.smk"


# Final data products. The training_waveforms rule is only defined when
# pregenerate_training_waveforms is True, so request it only in that case.
targets = [
    *rules.aggregate_val_waveforms.output,
    *rules.aggregate_testing_waveforms.output,
]
if config.get("pregenerate_training_waveforms", False):
    targets += rules.training_waveforms.output


rule all:
    default_target: True
    input:
        targets,
