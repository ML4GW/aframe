"""Top-level Snakefile

Run from a run directory made by `aframe-init snakemake`, whose run.sh
calls, from that directory:

    snakemake --snakefile <repo>/Snakefile --configfile config.yaml \
        --profile <repo>/pipeline/profiles/ldg

Each run has its own .snakemake/ directory and run_dir defaults to the
working directory. Settings that the run's config doesn't give come from
pipeline/config/config.yaml. A relative train_config is relative to the
repo, like the presets in pipeline/config/.
"""

import os
from pathlib import Path

REPO = Path(workflow.basedir)


configfile: str(REPO / "pipeline" / "config" / "config.yaml")


# `--config key=false` arrives as the string "false", which is truthy
for key, value in config.items():
    if isinstance(value, str) and value.lower() in ("true", "false"):
        config[key] = value.lower() == "true"
if config["run_dir"] is None:
    # Prevent runs from writing into the repo
    if Path.cwd().resolve() == REPO.resolve():
        raise WorkflowError(
            "Set run_dir (--config run_dir=...) or run from a run "
            "directory made by `aframe-init snakemake`"
        )
    config["run_dir"] = os.getcwd()
config["train_config"] = str(REPO / config["train_config"])

config.setdefault("background_dir", str(Path(config["run_dir"]) / "data"))
config.setdefault("waveforms_dir", str(Path(config["run_dir"]) / "waveforms"))
config.setdefault("log_dir", str(Path(config["run_dir"]) / "logs"))

run_dir = Path(config["run_dir"])
log_dir = Path(config["log_dir"])


include: "pipeline/resources.smk"
include: "projects/data/data.smk"
include: "projects/train/train.smk"
include: "projects/export/export.smk"
include: "projects/infer/infer.smk"
include: "projects/plots/plots.smk"


# Outside of `onstart` so that dry-runs do the check
check_gpus()


onstart:
    set_container_binds()
    check_images()
    check_triton_image()


rule all:
    default_target: True
    input:
        # stop_triton only exists in triton mode
        *([rules.stop_triton.output] if INFERENCE_MODE == "triton" else []),
        rules.sensitive_volume.output,
