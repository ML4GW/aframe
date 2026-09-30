"""Snakemake rules for model training.

Rules:
  train         run `train fit` locally via the Lightning CLI
  train_remote  submit training to Nautilus via train-remote

Only one of the two is defined, controlled by the `remote_train`
config flag.

For remote training, the model and batch file end up in
`remote_run_dir` on S3, so the rule's local output is a
sentinel file marking the rule's completion.

The train_config YAML defines all the model/data/trainer hyperparameters.
This rule adds the preprocessing args shared with inference.
"""

train_out = run_dir / "train"
train_log_dir = log_dir / "train"

TRAIN_CONTAINER = container("train")


# GPUs for training. Uses `train_num_gpus` of them, determined by `train_gpus`
# on a shared node or the least used with `auto`. `null` uses whatever is
# visible, which for slurm is the allocation.
TRAIN_GPU_ENV = gpu_env(config["train_gpus"], config["train_num_gpus"])
TRAIN_NUM_GPUS = config["train_num_gpus"]


def _train_waveform_inputs(wildcards):
    """Pre-generated training waveforms, when enabled."""
    if config["pregenerate_training_waveforms"]:
        return [str(train_waveforms / "training_waveforms.hdf5")]
    return []


# Arguments shared between local and remote training.
train_data_params = dict(
    train_config=config["train_config"],
    seed=config["seed"],
    ifos="[" + ",".join(config["ifos"]) + "]",
    sample_rate=config["sample_rate"],
    kernel_length=config["kernel_length"],
    fduration=config["fduration"],
    fftlength=config["fftlength"] or "null",
    highpass=config["highpass"],
    lowpass=config["lowpass"] or "null",
)

train_cli_args = (
    " --config {params.train_config}"
    " --seed_everything {params.seed}"
    " --data.ifos '{params.ifos}'"
    " --data.background_dir {params.background_dir}"
    " --data.waveforms_dir {params.waveforms_dir}"
    " --data.sample_rate {params.sample_rate}"
    " --data.kernel_length {params.kernel_length}"
    " --data.fduration {params.fduration}"
    " --data.fftlength {params.fftlength}"
    " --data.highpass {params.highpass}"
    " --data.lowpass {params.lowpass}"
    " --trainer.logger.save_dir {params.save_dir}"
)


if config["remote_train"]:

    rule train_remote:
        """Submit training to Nautilus and wait for the pod.

        train-remote runs on the submit node, while the training
        itself runs in a pod on the Nautilus. The background_dir,
        waveforms_dir, and remote_run_dir must all be s3:// paths that
        the pod can reach, and AWS_* / WANDB_API_KEY must be set in the
        submitting environment.

        NOTE: not yet functional
        """
        input:
            background=train_background_files,
            val_waveforms=str(train_waveforms / "val_waveforms.hdf5"),
            train_waveforms=_train_waveform_inputs,
        output:
            touch(str(train_out / "remote_train.done")),
        log:
            str(train_log_dir / "train_remote.log"),
        localrule: True
        params:
            **train_data_params,
            background_dir=config["remote_background_dir"],
            waveforms_dir=config["remote_waveforms_dir"],
            save_dir=config["remote_run_dir"],
        shell:
            "train-remote --train_args fit" + train_cli_args + " &> {log}"

else:

    rule train:
        """Train the aframe model with the Lightning CLI.

        With gpu_rules_local set to True, this runs on the node where
        snakemake is invoked.

        AFRAME_TRAIN_WAVEFORMS_DIR is exported so the config resolves
        to this run's validation/training waveform files. The GPU count
        overrides trainer.devices in the train config.
        """
        input:
            background=train_background_files,
            val_waveforms=str(train_waveforms / "val_waveforms.hdf5"),
            train_waveforms=_train_waveform_inputs,
        output:
            exported=str(train_out / "model_exported.pt2"),
            batch=str(train_out / "batch.hdf5"),
        log:
            str(train_log_dir / "train.log"),
        localrule: config["gpu_rules_local"]
        container:
            TRAIN_CONTAINER
        # never reaches condor, so slurm GPU keys only
        resources:
            **rule_resources("train"),
            slurm_partition=config["train_partition"],
            gpu=TRAIN_NUM_GPUS,
        params:
            **train_data_params,
            gpu_env=TRAIN_GPU_ENV,
            num_gpus=TRAIN_NUM_GPUS,
            background_dir=str(train_bg),
            waveforms_dir=str(train_waveforms),
            save_dir=str(train_out),
        shell:
            "{params.gpu_env}AFRAME_TRAIN_WAVEFORMS_DIR={params.waveforms_dir}"
            " python -m train fit"
            + train_cli_args
            + " --trainer.devices {params.num_gpus} &> {log}"
