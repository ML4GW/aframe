#!/usr/bin/env python3

import shutil
from pathlib import Path
from textwrap import dedent

from jsonargparse import ArgumentParser

root = Path(__file__).resolve().parent.parent

ONLINE_CONFIGS = [
    root / "projects" / "online" / "config.yaml",
    root / "projects" / "online" / "prior.yaml",
    root / "projects" / "online" / "crontab",
]


def copy_configs(path: Path, configs: list[Path]):
    """
    Copy the configuration files to the specified directory for editing.

    Args:
        path:
            The directory to copy the configuration files to.
        configs:
            The list of configuration files to copy.
    """

    for config in configs:
        shutil.copy(config, path / config.name)


def write_content(content: str, path: Path):
    content = dedent(content).strip("\n")
    with open(path, "w") as f:
        f.write(content)

    # make the file executable
    path.chmod(0o755)
    return content


def create_snakemake_runfile(path: Path, profile: str):
    config = path / "config.yaml"
    cmd = f"snakemake --configfile {config} --profile {profile}"
    content = f"""
    #!/bin/bash
    # Set AFRAME_DEV=1 to bind the working tree into containers.
    cd {root}
    source pipeline/.env
    {cmd}
    """
    runfile = path / "run.sh"
    write_content(content, runfile)


def create_online_runfile(path: Path):
    cmd = "apptainer run --nv "
    # bind /local/aframe for finding scitokens
    cmd += "--bind /local/aframe.online,$ONLINE_DATADIR"
    cmd += "--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "
    cmd += "--env AFRAME_ONLINE_OUTDIR=$AFRAME_ONLINE_OUTDIR "
    cmd += "--env ONLINE_DATADIR=$ONLINE_DATADIR "
    cmd += "--env AFRAME_WEIGHTS=$AFRAME_WEIGHTS "
    cmd += "--env AMPLFI_WEIGHTS=$AMPLFI_WEIGHTS "
    cmd += "--env BEARER_TOKEN_FILE=$BEARER_TOKEN_FILE "
    cmd += "--env SCITOKEN_FILE=$SCITOKEN_FILE "
    cmd += "$AFRAME_CONTAINER/online.sif /opt/env/bin/online "
    cmd += "--config $config 2>> monitoring.log"

    monitor_cmd = "apptainer run "
    monitor_cmd += f" --bind {path} "
    monitor_cmd += "$AFRAME_CONTAINER/online.sif /opt/env/bin/monitor "
    monitor_cmd += f"--run_dir {path} --out_dir $MONITOR_OUTDIR "
    monitor_cmd += ">> summary_pages.log 2>&1 &"

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
    export BEARER_TOKEN_FILE=/local/aframe.online/scitoken
    export SCITOKEN_FILE=/local/aframe.online/scitoken

    export AFRAME_KEYTAB=/home/aframe.online/robot/aframe-online_robot_aframe.ldas.cit.keytab
    export AFRAME_CREDKEY=aframe-online/robot/aframe.ldas.cit

    export RUN_DIR={path}

    # trained model weights
    export AMPLFI_HL_WEIGHTS=$RUN_DIR/models/amplfi-hl.ckpt
    export AMPLFI_HLV_WEIGHTS=$RUN_DIR/models/amplfi-hl.ckpt
    export AFRAME_WEIGHTS=$RUN_DIR/models/aframe.pt

    # file containing timeslide events detected
    # by a model with the AFRAME_WEIGHTS above
    export ONLINE_BACKGROUND_FILE=$RUN_DIR/data/background.hdf5
    # file containing detected events from an
    # injected campaign using AFRAME_WEIGHTS
    export ONLINE_FOREGROUND_FILE=$RUN_DIR/data/foreground.hdf5
    # file containing events that were rejected
    # during the injection simulation process
    export ONLINE_REJECTED_FILE=$RUN_DIR/data/rejected-parameters.hdf5

    # location where low latency data
    # is streamed, typically /kafka
    export ONLINE_DATADIR=/kafka/

    # where results and deployment logs will be writen
    export AFRAME_ONLINE_OUTDIR=$RUN_DIR/output

    # Location of Aframe containers
    export AFRAME_CONTAINER=$HOME/images/aframe/online.sif

    config=$RUN_DIR/config.yaml

    # Fill out and uncomment the following to perform monitoring
    # export MONITOR_OUTDIR=$HOME/public_html/{path.stem}
    # {monitor_cmd}

    export CUDA_VISIBLE_DEVICES=
    crash_count=0
    until {cmd}; do
        ((crash_count++))
        echo "Online deployment crashed on $(date) with error code $?,
        crash count = $crash_count" >> monitoring.log
        sleep 1
    done
    """  # noqa E501
    runfile = path / "run.sh"
    write_content(content, runfile)


def main():
    # snakemake subcommand
    snakemake_parser = ArgumentParser()
    snakemake_parser.add_argument(
        "-d", "--directory", type=Path, required=True
    )
    snakemake_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default="pipeline/profiles/ldg",
        help="Path to the snakemake profile directory",
    )

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
    subcommands.add_subcommand("snakemake", snakemake_parser)
    subcommands.add_subcommand("online", online_parser)

    args = parser.parse_args()
    subcommand = args.subcommand
    args = getattr(args, args.subcommand)
    directory = args.directory.resolve()
    weights_dir = getattr(args, "weights_dir", None)
    if weights_dir:
        weights_dir = weights_dir.resolve()

    # Create the run directory and move in weights if specified
    directory.mkdir(parents=True, exist_ok=True)
    if weights_dir:
        shutil.copytree(weights_dir, directory / "training")

    if subcommand == "snakemake":
        train_yaml = root / "projects" / "train" / "train.yaml"
        shutil.copy(train_yaml, directory / "train.yaml")
        run_config = directory / "config.yaml"
        run_config.write_text(
            f"# Overrides for pipeline/config/config.yaml.\n"
            f"run_dir: {directory}\n"
            f"train_config: {directory / 'train.yaml'}\n"
        )
        create_snakemake_runfile(directory, args.profile)

    elif subcommand == "online":
        copy_configs(directory, ONLINE_CONFIGS)
        create_online_runfile(directory)


if __name__ == "__main__":
    main()
