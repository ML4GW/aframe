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

import os

train_out = run_dir / "train"
train_log_dir = log_dir / "train"

TRAIN_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "train.sif")


def _train_waveform_inputs(wildcards):
    """Pre-generated training waveforms, when enabled."""
    if config.get("pregenerate_training_waveforms", False):
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
    fftlength=config.get("fftlength") or "null",
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


if config.get("remote_train", False):

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
            background=get_train_background_files,
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
        to this run's validation/training waveform files.
        """
        input:
            background=get_train_background_files,
            val_waveforms=str(train_waveforms / "val_waveforms.hdf5"),
            train_waveforms=_train_waveform_inputs,
        output:
            exported=str(train_out / "model_exported.pt2"),
            batch=str(train_out / "batch.hdf5"),
        log:
            str(train_log_dir / "train.log"),
        localrule: config.get("gpu_rules_local", True)
        container:
            TRAIN_CONTAINER
        resources:
            slurm_partition=config.get("train_partition", "gpuA40x4"),
            gpu=config.get("train_num_gpus", 1),
            mem_mb=config.get("train_mem_mb", 32000),
            runtime=2880,
        params:
            **train_data_params,
            background_dir=str(train_bg),
            waveforms_dir=str(train_waveforms),
            save_dir=str(train_out),
        shell:
            "AFRAME_TRAIN_WAVEFORMS_DIR={params.waveforms_dir}"
            " python -m train fit" + train_cli_args + " &> {log}"
