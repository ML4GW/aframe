"""Snakemake rule for model export.

Rules:
  export: compile the trained model into an accelerated format

The export_config YAML specifies all static export parameters.
Weights, batch_file, and repository_directory are determined here.
"""

import os

export_out = run_dir / "export"
export_log_dir = log_dir / "export"

EXPORT_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "export.sif")

remote_train = config.get("remote_train", False)


def _train_artifacts(wildcards):
    if remote_train:
        return [str(train_out / "remote_train.done")]
    return [str(train_out / "model.pt"), str(train_out / "batch.hdf5")]


rule export:
    """Export the trained model to an accelerated format.

With gpu_rules_local set to True, this runs on the node where
snakemake is invoked.
"""
    input:
        _train_artifacts,
    output:
        model_repo=directory(str(export_out / "model_repo")),
    log:
        str(export_log_dir / "export.log"),
    localrule: config.get("gpu_rules_local", True)
    container:
        EXPORT_CONTAINER
    params:
        export_config=config["export_config"],
        weights=(
            (config["remote_run_dir"] + "/model.pt")
            if remote_train
            else str(train_out / "model.pt")
        ),
        batch_file=(
            (config["remote_run_dir"] + "/batch.hdf5")
            if remote_train
            else str(train_out / "batch.hdf5")
        ),
    shell:
        "python -m export"
        " --config {params.export_config}"
        " --weights {params.weights}"
        " --batch_file {params.batch_file}"
        " --repository_directory {output.model_repo}"
        " &> {log}"
