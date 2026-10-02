"""Snakemake rule for model export.

Rules:
  export: compile the trained model into an accelerated format
"""

export_out = run_dir / "export"
export_log_dir = log_dir / "export"

EXPORT_CONTAINER = container("export")

# Remote training leaves the model and batch file on S3, and only a
# sentinel here
if config["remote_train"]:
    _train_artifacts = {"done": str(train_out / "remote_train.done")}
    _weights = config["remote_run_dir"] + "/model_exported.pt2"
    _batch_file = config["remote_run_dir"] + "/batch.hdf5"
else:
    _train_artifacts = {
        "weights": str(train_out / "model_exported.pt2"),
        "batch_file": str(train_out / "batch.hdf5"),
    }
    _weights = _train_artifacts["weights"]
    _batch_file = _train_artifacts["batch_file"]


rule export:
    """Export the trained model to an accelerated format.

With gpu_rules_local set to True, this runs on the node where
snakemake is invoked.
"""
    input:
        **_train_artifacts,
    output:
        model_repo=directory(str(export_out / "model_repo")),
    log:
        str(export_log_dir / "export.log"),
    localrule: config["gpu_rules_local"]
    container:
        EXPORT_CONTAINER
    # never reaches condor, so slurm GPU keys only
    resources:
        **rule_resources("export", "export"),
        slurm_partition=config["inference_partition"],
        gpu=1,
    params:
        preprocessor="utils.preprocessing.BatchWhitener",
        num_ifos=len(config["ifos"]),
        kernel_length=config["kernel_length"],
        sample_rate=config["sample_rate"],
        inference_sampling_rate=config["inference_sampling_rate"],
        batch_size=config["inference_batch_size"],
        fduration=config["fduration"],
        fftlength=config["fftlength"] or "null",
        psd_length=config["psd_length"],
        highpass=config["highpass"],
        lowpass=config["lowpass"] or "null",
        streams_per_gpu=config["streams_per_gpu"],
        gpu_env=gpu_env(config["inference_gpus"], 1),
        weights=_weights,
        batch_file=_batch_file,
    shell:
        "{params.gpu_env}python -m export"
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
        # the rest of the preprocessor's arguments are linked from the above
        " --preprocessor.init_args.fftlength {params.fftlength}"
        " --preprocessor.init_args.highpass {params.highpass}"
        " --preprocessor.init_args.lowpass {params.lowpass}"
        " &> {log}"
