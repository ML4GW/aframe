import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.s3 import open_file


def test_open_local_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        temp_file = f.name
        f.write("test contents")

    try:
        with open_file(temp_file, mode="r") as f:
            content = f.read()
        assert content == "test contents"
    finally:
        Path(temp_file).unlink()


def test_open_local_file_binary():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        temp_file = f.name
        f.write(b"test binary contents")

    try:
        with open_file(temp_file, mode="rb") as f:
            content = f.read()
        assert content == b"test binary contents"
    finally:
        Path(temp_file).unlink()


@patch("s3fs.S3FileSystem")
def test_open_s3_file(mock_s3fs):
    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_fs.open.return_value = mock_file
    mock_s3fs.return_value = mock_fs

    result = open_file("s3://bucket/file.txt", mode="r")

    mock_fs.open.assert_called_once_with("s3://bucket/file.txt", "r")
    assert result == mock_file
