"""Top-level Snakefile

Run from a run directory made by `aframe-init snakemake`, whose run.sh
copies the local repo into the run's code/ directory and then calls, from
the run directory:

    snakemake --snakefile code/Snakefile \
        --configfiles code/pipeline/config/config.yaml config.yaml \
        --profile code/pipeline/profiles/ldg

Each run has its own .snakemake/ directory, and runs its own copy of the
code, so edits to the local repo take effect when a run is restarted.
The defaults in pipeline/config/config.yaml come first, so the run's
config.yaml overrides them, and `--config` overrides both. They're given
on the command line rather than with `configfile:` because the htcondor
executor causes the default config to be merged after the run's config,
overriding the run-specific settings. A relative train_config is
relative to the repo, like the presets in pipeline/config/.

Every path that the rules use is relative to the run directory, so that a
condor job that shares no filesystem with the submit node can recreate
them in its scratch directory.
"""

from pathlib import Path

REPO = Path(workflow.basedir)

# `--config key=false` arrives as the string "false", which is truthy
for key, value in config.items():
    if isinstance(value, str) and value.lower() in ("true", "false"):
        config[key] = value.lower() == "true"

# Prevent runs from writing into the repo
if Path.cwd().resolve() == REPO.resolve():
    raise WorkflowError(
        "Run from a run directory made by `aframe-init snakemake`, not the repo"
    )
if config["run_dir"] and Path(config["run_dir"]).resolve() != Path.cwd().resolve():
    raise WorkflowError(
        f"Run snakemake from run_dir ({config['run_dir']}), since every path is "
        "relative to it"
    )
config["train_config"] = str(REPO / config["train_config"])

run_dir = Path(".")
log_dir = Path("logs")


include: "pipeline/resources.smk"


# Background and waveforms are always in the run's data/ and waveforms/.
# background_dir and waveforms_dir name another run's directories whose
# files to reuse instead of fetching or generating them again. Their files
# are linked into this run's directories, on the AP only.
for key, name in (("background_dir", "data"), ("waveforms_dir", "waveforms")):
    if config.get(key) and not workflow.remote_exec:
        link_files(config[key], name)


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
