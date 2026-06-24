"""Snakemake rule for model export.

Rules:
  export: compile the trained model into an accelerated format
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
        preprocessor=config["export_preprocessor"],
        num_ifos=len(config["ifos"]),
        kernel_length=config["kernel_length"],
        sample_rate=config["sample_rate"],
        inference_sampling_rate=config["inference_sampling_rate"],
        batch_size=config["inference_batch_size"],
        fduration=config["fduration"],
        fftlength=config.get("fftlength") or "null",
        psd_length=config["psd_length"],
        highpass=config["highpass"],
        streams_per_gpu=config["streams_per_gpu"],
        weights=(
            (config["remote_run_dir"] + "/model_exported.pt2")
            if remote_train
            else str(train_out / "model_exported.pt2")
        ),
        batch_file=(
            (config["remote_run_dir"] + "/batch.hdf5")
            if remote_train
            else str(train_out / "batch.hdf5")
        ),
    shell:
        "python -m export"
        " --weights {params.weights}"
        " --batch_file {params.batch_file}"
        " --repository_directory {output.model_repo}"
        " --preprocessor {params.preprocessor}"
        " --num_ifos {params.num_ifos}"
        " --kernel_length {params.kernel_length}"
        " --sample_rate {params.sample_rate}"
        " --inference_sampling_rate {params.inference_sampling_rate}"
        " --batch_size {params.batch_size}"
        " --fduration {params.fduration}"
        " --psd_length {params.psd_length}"
        " --streams_per_gpu {params.streams_per_gpu}"
        " --preprocessor.init_args.kernel_length {params.kernel_length}"
        " --preprocessor.init_args.sample_rate {params.sample_rate}"
        " --preprocessor.init_args.inference_sampling_rate"
        " {params.inference_sampling_rate}"
        " --preprocessor.init_args.batch_size {params.batch_size}"
        " --preprocessor.init_args.fduration {params.fduration}"
        " --preprocessor.init_args.fftlength {params.fftlength}"
        " --preprocessor.init_args.highpass {params.highpass}"
        " &> {log}"
