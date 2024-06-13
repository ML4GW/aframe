#!/usr/bin/env python3

import configparser
import shutil
from pathlib import Path
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
        # of the config files in the init dir
        if config.suffix == ".cfg" and config.name != "base.cfg":
            cfg = configparser.ConfigParser()
            cfg.read(config)
            cfg["core"]["inherit"] = str(path / "base.cfg")

            train_task = (
                "luigi_Train" if pipeline == "sandbox" else "luigi_TuneRemote"
            )
            cfg[train_task]["config"] = str(path / "config.yaml")

            if s3_bucket is not None:
                cfg["luigi_Train"]["train_remote"] = "true"

            with open(dest, "w") as f:
                cfg.write(f)
        else:
            shutil.copy(config, dest)


def create_env(path: Path, s3_bucket: Optional[Path] = None):
    # if s3 bucket is provided
    # store training data and training info there
    base = path if s3_bucket is None else s3_bucket
    content = f"""
# Export environment variables
export AFRAME_TRAIN_DATA_DIR={base}/data/train
export AFRAME_TEST_DATA_DIR={path}/data/test
export AFRAME_TRAIN_RUN_DIR={base}/training
export AFRAME_CONDOR_DIR={path}/condor
export AFRAME_RESULTS_DIR={path}/results
export AFRAME_TMPDIR={path}/tmp/
"""

    env = path / "env.env"
    with open(env, "w") as f:
        f.write(content)


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
        args.s3_bucket,
    )
    create_env(directory, args.s3_bucket)


if __name__ == "__main__":
    main()
