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
    root / "projects" / "train" / "config.yaml",
    root / "projects" / "train" / "configs" / "search_space.py",
]

SANDBOX_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "bbh.cfg",
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "base.cfg",
    root / "projects" / "train" / "config.yaml",
]

ONLINE_CONFIGS = [
    root / "projects" / "online" / "config.yaml",
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

    path.mkdir(parents=True, exist_ok=True)
    for config in configs:
        dest = path / config.name
        # update the luigi/law config file to point to the paths
        # of other relevant config files in the init dir
        if config.suffix == ".cfg" and config.name != "base.cfg":
            dest = path / f"{pipeline}.cfg"
            cfg = configparser.ConfigParser()
            cfg.read(config)
            cfg["core"]["inherit"] = str(path / "base.cfg")

            # set the train config file
            # to the one in the init directory
            train_task = (
                "luigi_Train" if pipeline == "sandbox" else "luigi_TuneRemote"
            )
            cfg[train_task]["config"] = str(path / "config.yaml")

            # set the search space for the tune pipeline
            # to the search space file in the init directory
            if pipeline == "tune":
                cfg[train_task]["search_space"] = str(path / "search_space.py")

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
    # bind XDG_RUNTIME_DIR for finding scitokens
    cmd += "--bind $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR "
    cmd += "--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "
    cmd += "--env AFRAME_ONLINE_OUTDIR=$AFRAME_ONLINE_OUTDIR "
    cmd += "--env ONLINE_DATADIR=$ONLINE_DATADIR "
    cmd += "--env AFRAME_WEIGHTS=$AFRAME_WEIGHTS "
    cmd += "--env AMPLFI_WEIGHTS=$AMPLFI_WEIGHTS "
    cmd += "$AFRAME_CONTAINER_ROOT/online.sif /opt/env/bin/online "
    cmd += "--config $config 2>> monitoring.log"

    content = f"""
    #!/bin/bash
    # trained model weights
    export AMPLFI_WEIGHTS=
    export AFRAME_WEIGHTS=

    # file containing timeslide events detected
    # by a model with the AFRAME_WEIGHTS above
    export ONLINE_BACKGROUND_FILE=

    # location where low latency data
    # is streamed, typically /dev/shm/kakfka
    export ONLINE_DATADIR=/dev/shm/kafka/

    # where results and deployment logs will be writen
    export AFRAME_ONLINE_OUTDIR={path}

    config={path}/config.yaml

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
    # make the below one string
    cmd = f"LAW_CONFIG_FILE={config} poetry run --directory {root} "
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
        choices=["sandbox", "tune"],
        default="sandbox",
        help="Specify whether this is a sandbox or tune run",
    )
    offline_parser.add_argument("-d", "--directory", type=Path, required=True)
    offline_parser.add_argument("--s3-bucket")

    # online subcommand
    online_parser = ArgumentParser()
    online_parser.add_argument("-d", "--directory", type=Path, required=True)

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

    if subcommand == "offline":
        if args.s3_bucket is not None and not args.s3_bucket.startswith(
            "s3://"
        ):
            raise ValueError(
                "S3 bucket must be in the format s3://{bucket-name}/"
            )
        configs = TUNE_CONFIGS if args.mode == "tune" else SANDBOX_CONFIGS
        copy_configs(directory, configs, args.mode)
        create_offline_runfile(directory, args.mode, args.s3_bucket)

    elif subcommand == "online":
        copy_configs(directory, ONLINE_CONFIGS, "online")
        create_online_runfile(directory)


if __name__ == "__main__":
    main()
