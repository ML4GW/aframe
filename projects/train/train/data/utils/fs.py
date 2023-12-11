import logging
import os
from tempfile import mkdtemp

import ray
import s3fs
from botocore.exceptions import ResponseStreamingError


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
        # we have remote data, but we didn't explicitly
        # specify a directory to download it to, so create
        # a tmp directory using the worker id so that each
        # worker process downloads its own copy of the data
        # only on its first training run
        try:
            worker_id = ray.get_runtime_context().get_worker_id()
        except Exception:  # TODO: what's the exact exception?
            # we haven't initialized ray, so just create
            # a random temporary directory to download to
            data_dir = mkdtemp()
        else:
            data_dir = f"/tmp/{worker_id}"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _download(
    s3: s3fs.S3FileSystem, source: str, target: str, num_retries: int = 3
):
    """
    Cheap wrapper around s3.get to try to avoid issues
    from interrupted reads.
    """
    for i in range(num_retries):
        try:
            s3.get(source, target)
            break
        except ResponseStreamingError:
            logging.info(
                "Download attempt {} for object {} "
                "was interrupted, retrying".format(i + 1, source)
            )
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

    # download background
    for f in background_fnames:
        target = data_dir + f.replace(f"{bucket}", "")
        if not os.path.exists(target):
            logging.info(f"Downloading {f} to {target}")
            _download(s3, f, target)
        else:
            logging.info(f"Object {f} already downloaded")

    # now download our signal data
    path = "signals.hdf5"
    target = f"{data_dir}/{path}"
    if not os.path.exists(target):
        logging.info(f"Downloading {path} to {target}")
        _download(s3, f"{bucket}/{path}", target)
    else:
        logging.info(f"Object {path} already downloaded")
    logging.info("Data download complete")
