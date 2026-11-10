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
    root / "projects" / "train" / "configs" / "training_prior.yaml",
]

REVIEW_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "review.cfg"
]

ONLINE_CONFIGS = [
    root / "projects" / "online" / "config.yaml",
    root / "projects" / "online" / "prior.yaml",
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
            cfg[train_task]["train_config"] = str(path / "train.yaml")

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
            cfg["luigi_ExportLocal"]["export_config"] = str(
                path / "export.yaml"
            )
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
    # bind TOKEN_DIR for finding scitokens
    cmd += "--bind $TOKEN_DIR,$ONLINE_DATADIR "
    cmd += "--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "
    cmd += "--env AFRAME_ONLINE_OUTDIR=$AFRAME_ONLINE_OUTDIR "
    cmd += "--env ONLINE_DATADIR=$ONLINE_DATADIR "
    cmd += "--env ARRAKIS_SERVER=$ARRAKIS_SERVER "
    cmd += "--env AFRAME_WEIGHTS=$AFRAME_WEIGHTS "
    cmd += "--env AMPLFI_WEIGHTS=$AMPLFI_WEIGHTS "
    cmd += "--env BEARER_TOKEN_FILE=$BEARER_TOKEN_FILE "
    cmd += "--env SCITOKEN_FILE=$SCITOKEN_FILE "
    cmd += "$AFRAME_CONTAINER /opt/env/bin/online "
    cmd += "--config $config 2>> monitoring.log"

    monitor_cmd = "apptainer run "
    monitor_cmd += f" --bind {path} "
    monitor_cmd += "$AFRAME_CONTAINER /opt/env/bin/monitor "
    monitor_cmd += f"--run_dir {path} --out_dir $MONITOR_OUTDIR "
    monitor_cmd += ">> summary_pages.log 2>&1"
    # restart the monitor if it dies, the same way the search is restarted
    monitor_cmd = (
        "(until "
        + monitor_cmd
        + '; do echo "Monitor crashed on $(date), restarting"'
        " >> summary_pages.log; sleep 1; done) &"
    )

    content = f"""
    #!/bin/bash

    # Stop the search and monitor together. The search runs in its own
    # process group and the monitor runs in this script's group, so both
    # have to be handled.
    stop() {{
        # Ignore any further signals so that this script isn't killed by
        # `kill -TERM 0` and a second Ctrl-C doesn't interrupt the shutdown
        trap '' INT TERM HUP EXIT

        # Remove the file containing the PID of this script
        rm -f {path}/run.pid

        # Send a TERM signal to everything. The search will clean up
        # before it exits. The subprocesses and monitor will halt
        # immediately.
        kill -TERM -- -$search_pid 2>/dev/null
        kill -TERM 0

        # Give the search up to GRACE seconds to finish cleaning up,
        # and then kill whatever is left, including this script
        for _ in $(seq $GRACE); do
            # The -0 checks if the process is still running
            kill -0 $search_pid 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- -$search_pid 2>/dev/null
        kill -KILL 0
    }}
    trap stop INT TERM HUP EXIT
    # Seconds the search gets to shut down cleanly before it's killed
    GRACE=30

    # Record this script's PID, so the run can be stopped with
    # `kill $(cat {path}/run.pid)`
    echo $$ > {path}/run.pid
    echo "run.sh started on $(date) with PID $$" >> {path}/monitoring.log

    # Space-separated addresses to email if the search keeps crashing
    ALERT_EMAILS=
    # Email once the search has crashed this many times in a row
    ALERT_AFTER=5
    # Longest wait between restarts, in seconds. A run that lasted
    # longer than this is treated as a separate crash group.
    MAX_DELAY=300

    # ligo skymap from samples
    export TQDM_DISABLE=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    # scitoken auth
    # it is recommended not to store token in /home;
    # use the expected $XDG_RUNTIME_DIR if the variable
    # is set, falling back to /tmp
    export TOKEN_DIR=${{XDG_RUNTIME_DIR:-/tmp}}
    export BEARER_TOKEN_FILE=$TOKEN_DIR/bt_u$(id -u)
    export SCITOKEN_FILE=$BEARER_TOKEN_FILE

    export AFRAME_KEYTAB=/home/aframe.online/robot/aframe-online_robot_aframe.ldas.cit.keytab
    export AFRAME_CREDKEY=aframe-online/robot/aframe.ldas.cit

    export RUN_DIR={path}

    # trained model weights
    export AMPLFI_HL_WEIGHTS=$RUN_DIR/models/amplfi-hl.ckpt
    export AMPLFI_HLV_WEIGHTS=$RUN_DIR/models/amplfi-hlv.ckpt
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
    # Used when data_source="frames"
    export ONLINE_DATADIR=/kafka/

    # Used when data_source="arrakis"
    export ARRAKIS_SERVER=grpc://arrakis:31206

    # where results and deployment logs will be writen
    export AFRAME_ONLINE_OUTDIR=$RUN_DIR/output

    # Location of Aframe containers
    export AFRAME_CONTAINER=$HOME/images/aframe/online.sif

    config=$RUN_DIR/config.yaml

    # Fill out and uncomment the following to perform monitoring
    # export MONITOR_OUTDIR=$HOME/public_html/{path.stem}
    # {monitor_cmd}

    export CUDA_VISIBLE_DEVICES=

    # Restart the search whenever it crashes, waiting longer after each
    # crash in a row to avoid repeatedly restarting, e.g., when resources
    # aren't ready after a server reboot.
    crash_count=0
    delay=1
    started=$SECONDS

    # Start the process in the background and store its PID in search_pid.
    # Wait until that process returns, and then clean up, log, and
    # potentially send an email.
    until setsid {cmd} & search_pid=$!; wait $search_pid; do
        return_code=$?

        # Kill any lingering subprocesses
        kill -KILL -- -$search_pid 2>/dev/null

        # If its been long enough, consider a new crash to be part of a separate
        # group of crashes
        if (( SECONDS - started > MAX_DELAY )); then
            crash_count=0
            delay=1
        fi

        # Log the return code and the count of crashes
        crash_count=$(( crash_count + 1 ))
        echo "Online deployment crashed on $(date) with error code $return_code, crash $crash_count in a row" >> monitoring.log

        # If we've crashed enough times in a row and ALERT_EMAILS have been given, then send an email
        if (( crash_count == ALERT_AFTER )) && [[ -n $ALERT_EMAILS ]]; then
            tail -n 50 monitoring.log | mail -s "Aframe online search in $RUN_DIR has crashed $crash_count times in a row" $ALERT_EMAILS
        fi

        $ Sleep for delay seconds in the background so that an interrupt doesn't get delayed
        sleep $delay & wait $!
        # Double the value of delay unless that value is greater than MAX_DELAY, in which case set the value to MAX_DELAY
        delay=$(( delay * 2 > MAX_DELAY ? MAX_DELAY : delay * 2 ))
        started=$SECONDS
    done
    """  # noqa E501
    runfile = path / "run.sh"
    write_content(content, runfile)


def create_online_crontab(path: Path):
    content = f"""
    # Relaunch this run when the node reboots after giving it time to
    # finish starting up. Add the line below to the account's crontab
    # with `crontab -e`.
    @reboot sleep 120 && cd {path} && /bin/bash run.sh
    """
    crontab = path / "crontab"
    write_content(content, crontab)


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
    export AFRAME_TRAIN_BACKGROUND_DIR=
    export AFRAME_TRAIN_WAVEFORMS_DIR={base}/data/train
    export AFRAME_TEST_BACKGROUND_DIR=
    export AFRAME_TEST_WAVEFORMS_DIR={path}/data/test
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
        weights_dest = "models" if subcommand == "online" else "training"
        shutil.copytree(weights_dir, directory / weights_dest)

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
        create_online_crontab(directory)


if __name__ == "__main__":
    main()
