import s3fs


def open_file(path, mode="rb"):
    """
    Open a file from a local path or an S3 path
    """
    if str(path).startswith("s3://"):
        fs = s3fs.S3FileSystem()
        return fs.open(path, mode)
    else:
        # For local paths
        return open(path, mode)
