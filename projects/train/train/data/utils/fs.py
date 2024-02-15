import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tempfile import gettempdir

import s3fs
from botocore.exceptions import ClientError, ResponseStreamingError
from filelock import FileLock
from fsspec.exceptions import FSTimeoutError


def split_data_dir(data_dir: str):
    """
    Check if a data directory specifies a remote s3
    source by including  `"s3://"` at the start of
    the directory name.
    """
    if data_dir.startswith("s3://"):
        bucket = data_dir.replace("s3://", "")

        # check if specified a target location to map to
        # by adding a colon at the end of our bucket
        bucket, *data_dir = bucket.split(":")
        data_dir = data_dir[0] if data_dir else None
        return bucket, data_dir
    else:
        return None, data_dir


def get_data_dir(data_dir: str):
    # generate our local node data directory
    # if our specified data source is remote
    bucket, data_dir = split_data_dir(data_dir)
    if bucket is not None and data_dir is None:
        tmpdir = gettempdir()
        logging.info("Downloading data to local tmp directory")
        data_dir = f"{tmpdir}/data-tmp"

    logging.info(f"Downloading data to {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _download(
    s3: s3fs.S3FileSystem, source: str, target: str, num_retries: int = 5
):
    """
    Cheap wrapper around s3.get to try to avoid issues
    from interrupted reads.
    """

    lockfile = f"{target}.lock"
    if os.path.exists(target):
        logging.info(f"Object {source} already downloaded")
        return

    logging.info(f"Downloading {source} to {target}")
    for i in range(num_retries):
        try:
            with FileLock(lockfile):
                s3.get(source, target)
                break
        except (ResponseStreamingError, FSTimeoutError, ClientError):
            logging.info(
                "Download attempt {} for object {} "
                "was interrupted, retrying".format(i + 1, source)
            )
            time.sleep(5)
            try:
                os.remove(target)
            except FileNotFoundError:
                continue
    else:
        raise RuntimeError(
            "Failed to download object {} due to repeated "
            "connection interruptions".format(source)
        )


def download_training_data(bucket: str, data_dir: str):
    """
    Download s3 data if it doesn't exist.
    """
    logging.info(
        "Downloading data from S3 bucket {} to "
        "local directory {}".format(bucket, data_dir)
    )

    # make a local directory to cache data if it
    # doesn't already exist
    background_dir = f"{data_dir}/background"
    os.makedirs(background_dir, exist_ok=True)

    # check to make sure the specified bucket
    # actually has data to download
    s3 = s3fs.S3FileSystem()
    background_fnames = s3.glob(f"{bucket}/background/*.hdf5")
    if not background_fnames:
        raise ValueError(f"No background data at {bucket} to download")

    # multiprocess download of training data
    targets = [
        data_dir + f.replace(f"{bucket}", "") for f in background_fnames
    ]
    download = partial(_download, s3)
    path = "signals.hdf5"
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            download, f"{bucket}/{path}", f"{data_dir}/{path}"
        )
        executor.map(download, background_fnames, targets)

    future.result()
