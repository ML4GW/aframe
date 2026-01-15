import s3fs

from pathlib import Path


def open_file(path: str | Path, mode: str = "rb"):
    """
    Open a file from either local filesystem or S3 remote storage.

    Provides a single interface for opening files that automatically
    detects whether the path is local or remote S3 and returns the
    appropriate file object.

    Args:
        path (str or Path): Path to the file. Should start with 's3://' for S3
            paths, otherwise treated as local filesystem path.
        mode (str, optional): File opening mode (e.g., 'rb' for binary read,
            'r' for text read). Defaults to 'rb' (binary read).

    Returns:
        file object: File object from either s3fs (for S3 paths) or built-in
            open function (for local paths).

    Examples:
        >>> # Open local file in binary mode
        >>> with open_file('data.txt') as f:
        ...     data = f.read()

        >>> # Open S3 file in text mode
        >>> with open_file('s3://bucket/data.txt', mode='r') as f:
        ...     data = f.read()
    """
    # Check if path is S3 remote path
    if str(path).startswith("s3://"):
        # Use s3fs for remote S3 paths
        fs = s3fs.S3FileSystem()
        return fs.open(path, mode)
    else:
        # Use built-in open for local filesystem paths
        return open(path, mode)
