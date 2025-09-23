#!/usr/bin/env python3

import configparser
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Optional

from jsonargparse import ArgumentParser

root = Path(__file__).resolve().parent.parent
TUNE_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "tune.cfg",
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "base.cfg",
    root / "projects" / "train" / "train.yaml",
    root / "projects" / "train" / "configs" / "tune.yaml",
]

SANDBOX_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "bbh.cfg",
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "base.cfg",
    root / "projects" / "train" / "train.yaml",
    root / "projects" / "export" / "export.yaml",
]

REVIEW_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "review.cfg"
]

ONLINE_CONFIGS = [
    root / "projects" / "online" / "config.yaml",
    root / "projects" / "online" / "prior.yaml",
    root / "projects" / "online" / "crontab",
]


def copy_configs(
    path: Path,
    configs: list[Path],
    pipeline: str,
):
    """
    Copy the configuration files to the specified directory for editing.

    Any path specific configurations will be updated to point to the
    correct paths in the new directory.

    Args:
        path:
            The directory to copy the configuration files to.
        configs:
            The list of configuration files to copy.
        pipeline:
            The type of pipeline to initialize. Either 'tune' or 'sandbox'.
    """

    for config in configs:
        dest = path / config.name
        # update the luigi/law config file to point to the paths
        # of other relevant config files in the init dir
        if config.suffix == ".cfg" and config.name not in [
            "base.cfg",
            "review.cfg",
        ]:
            dest = path / f"{pipeline}.cfg"
            cfg = configparser.ConfigParser()
            cfg.read(config)
            cfg["core"]["inherit"] = str(path / "base.cfg")

            # set the train config file
            # to the one in the init directory
            train_task = (
                "luigi_Train" if pipeline == "sandbox" else "luigi_TuneTask"
            )
            cfg[train_task]["config"] = str(path / "train.yaml")

            # if tuning, set the tune config file
            if pipeline == "tune":
                cfg[train_task]["tune_config"] = str(path / "tune.yaml")

            with open(dest, "w") as f:
                cfg.write(f)
        elif config.name in ["base.cfg", "review.cfg"]:
            cfg = configparser.ConfigParser()
            # Need this to preserve case of keys
            cfg.optionxform = str
            cfg.read(config)
            cfg["luigi_ExportLocal"]["config"] = str(path / "export.yaml")
            with open(dest, "w") as f:
                cfg.write(f)
        else:
            shutil.copy(config, dest)


def write_content(content: str, path: Path):
    content = dedent(content).strip("\n")
    with open(path, "w") as f:
        f.write(content)

    # make the file executable
    path.chmod(0o755)
    return content


def create_online_runfile(path: Path):
    cmd = "apptainer run --nv "
    # bind /local/aframe for finding scitokens
    cmd += "--bind /local/aframe "
    cmd += "--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "
    cmd += "--env AFRAME_ONLINE_OUTDIR=$AFRAME_ONLINE_OUTDIR "
    cmd += "--env ONLINE_DATADIR=$ONLINE_DATADIR "
    cmd += "--env AFRAME_WEIGHTS=$AFRAME_WEIGHTS "
    cmd += "--env AMPLFI_WEIGHTS=$AMPLFI_WEIGHTS "
    cmd += "--env BEARER_TOKEN_FILE=$BEARER_TOKEN_FILE "
    cmd += "--env SCITOKEN_FILE=$SCITOKEN_FILE "
    cmd += "$AFRAME_CONTAINER_ROOT/online.sif /opt/env/bin/online "
    cmd += "--config $config 2>> monitoring.log"

    monitor_cmd = "apptainer run "
    monitor_cmd += f" --bind {path} "
    monitor_cmd += "$AFRAME_CONTAINER_ROOT/online.sif /opt/env/bin/monitor "
    monitor_cmd += f"--run_dir {path} --out_dir $MONITOR_OUTDIR &"

    content = f"""
    #!/bin/bash

    control_c() {{
        kill $$
        exit
    }}
    trap control_c SIGINT

    # ligo skymap from samples
    export TQDM_DISABLE=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    # scitoken auth
    # it is recommended not to store token
    # on /home/ filesystem: should be in
    # /local/$USER somewhere
    export BEARER_TOKEN_FILE=
    export SCITOKEN_FILE=

    # trained model weights
    export AMPLFI_HL_WEIGHTS=
    export AMPLFI_HLV_WEIGHTS=
    export AFRAME_WEIGHTS=

    # file containing timeslide events detected
    # by a model with the AFRAME_WEIGHTS above
    export ONLINE_BACKGROUND_FILE=
    # file containing detected events from an
    # injected campaign using AFRAME_WEIGHTS
    export ONLINE_FOREGROUND_FILE=
    # file containing events that were rejected
    # during the injection simulation process
    export ONLINE_REJECTED_FILE=

    # location where low latency data
    # is streamed, typically /dev/shm/kakfka
    export ONLINE_DATADIR=/dev/shm/kafka/

    # where results and deployment logs will be writen
    export AFRAME_ONLINE_OUTDIR={path}

    # Location of Aframe containers
    export AFRAME_CONTAINER_ROOT=

    config={path}/config.yaml

    # Fill out and uncomment the following to perform monitoring
    # export MONITOR_OUTDIR=
    # {monitor_cmd}

    export CUDA_VISIBLE_DEVICES=
    crash_count=0
    until {cmd}; do
        ((crash_count++))
        echo "Online deployment crashed on $(date) with error code $?,
        crash count = $crash_count" >> monitoring.log
        sleep 1
    done
    """
    runfile = path / "run.sh"
    write_content(content, runfile)


def create_offline_runfile(
    path: Path, pipeline: str, s3_bucket: Optional[Path] = None
):
    # if s3 bucket is provided
    # store training data and training info there
    base = path if s3_bucket is None else s3_bucket

    config = path / f"{pipeline}.cfg"
    # For running the review check, we're overloading the sandbox pipeline,
    # so reset the name
    if pipeline == "review":
        pipeline = "sandbox"
    # make the below one string
    cmd = f"LAW_CONFIG_FILE={config} uv run --directory {root} "
    cmd += f"law run aframe.pipelines.sandbox.{pipeline.capitalize()} "
    cmd += "--workers 5 --gpus 0"
    content = f"""
    #!/bin/bash
    # Export environment variables
    export AFRAME_TRAIN_DATA_DIR={base}/data/train
    export AFRAME_TEST_DATA_DIR={path}/data/test
    export AFRAME_TRAIN_RUN_DIR={base}/training
    export AFRAME_CONDOR_DIR={path}/condor
    export AFRAME_RESULTS_DIR={path}/results
    export AFRAME_TMPDIR={path}/tmp/

    # launch pipeline; modify the gpus, workers etc. to suit your needs
    # note that if you've made local code changes not in the containers
    # you'll need to add the --dev flag!
    {cmd}
    """

    runfile = path / "run.sh"
    write_content(content, runfile)


def main():
    # offline subcommand (sandbox or tune)
    offline_parser = ArgumentParser()
    offline_parser.add_argument(
        "--mode",
        choices=["sandbox", "tune", "review"],
        default="sandbox",
        help="Specify the type run to initialize",
    )
    offline_parser.add_argument("-d", "--directory", type=Path, required=True)
    offline_parser.add_argument("--s3-bucket")
    offline_parser.add_argument("--weights-dir", type=Path)

    # online subcommand
    online_parser = ArgumentParser()
    online_parser.add_argument("-d", "--directory", type=Path, required=True)
    online_parser.add_argument("--weights-dir", type=Path)

    # main parser
    parser = ArgumentParser(
        description="Initialize a directory with configuration files "
        "for running aframe offline and online pipelines."
    )
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("online", online_parser)
    subcommands.add_subcommand("offline", offline_parser)

    args = parser.parse_args()
    subcommand = args.subcommand
    args = getattr(args, args.subcommand)
    directory = args.directory.resolve()
    weights_dir = args.weights_dir.resolve() if args.weights_dir else None

    # Create the run directory and move in weights if specified
    directory.mkdir(parents=True, exist_ok=True)
    if weights_dir:
        shutil.copytree(weights_dir, directory / "training")

    if subcommand == "offline":
        if args.s3_bucket is not None and not args.s3_bucket.startswith(
            "s3://"
        ):
            raise ValueError(
                "S3 bucket must be in the format s3://{bucket-name}/"
            )
        if args.mode == "sandbox":
            configs = SANDBOX_CONFIGS
        elif args.mode == "tune":
            configs = TUNE_CONFIGS
        elif args.mode == "review":
            configs = REVIEW_CONFIGS
        else:
            raise ValueError("Mode must be 'sandbox', 'tune', or 'review'")
        copy_configs(directory, configs, args.mode)
        create_offline_runfile(directory, args.mode, args.s3_bucket)

    elif subcommand == "online":
        copy_configs(directory, ONLINE_CONFIGS, "online")
        create_online_runfile(directory)


if __name__ == "__main__":
    main()
