#!/usr/bin/env python3

import configparser
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Optional

import jsonargparse

root = Path(__file__).resolve().parent.parent
TUNE_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "tune.cfg",
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "base.cfg",
    root / "projects" / "train" / "config.yaml",
    root / "projects" / "train" / "train" / "tune" / "search_space.py",
]

SANDBOX_CONFIGS = [
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "bbh.cfg",
    root / "aframe" / "pipelines" / "sandbox" / "configs" / "base.cfg",
    root / "projects" / "train" / "config.yaml",
]


def copy_configs(
    path: Path,
    configs: list[Path],
    pipeline: str,
    s3_bucket: Optional[Path] = None,
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
        s3_bucket:
            The s3 bucket to store the data and
            training results in. Defaults to None.
    """

    path.mkdir(parents=True, exist_ok=True)
    for config in configs:
        dest = path / config.name
        # update the config file to point to the paths
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


def create_runfile(
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

    content = dedent(content).strip("\n")
    env = path / "run.sh"
    with open(env, "w") as f:
        f.write(content)

    # make the file executable
    env.chmod(0o755)


def main():
    parser = jsonargparse.ArgumentParser(
        description="Initialize a directory with configuration files "
        "for running aframe pipelines."
    )
    parser.add_argument(
        "pipeline",
        help="The type of pipeline to initialize. "
        "Either 'tune' or 'sandbox'. Default is sandbox",
        default="sandbox",
    )

    parser.add_argument(
        "-d",
        "--directory",
        help="The directory to initialize the aframe analysis in",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--s3-bucket",
        help="Location for storing the data and results in s3 "
        "if training remotely. If not provided, the data and "
        "results will be stored locally in the --directory argument",
        required=False,
    )

    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.pipeline not in ["tune", "sandbox"]:
        raise ValueError(
            "Invalid pipeline type. Must be either 'tune' or 'sandbox'."
        )

    if args.s3_bucket is not None and not args.s3_bucket.startswith("s3://"):
        raise ValueError("S3 bucket must be in the format s3://{bucket-name}/")

    copy_configs(
        directory,
        TUNE_CONFIGS if args.pipeline == "tune" else SANDBOX_CONFIGS,
        args.pipeline,
        args.s3_bucket,
    )
    create_runfile(directory, args.pipeline, args.s3_bucket)


if __name__ == "__main__":
    main()
