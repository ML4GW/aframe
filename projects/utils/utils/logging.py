import logging
import sys
from pathlib import Path
from typing import Optional, Union


def configure_logging(
    filename: Optional[Union[str, Path]] = None, verbose: bool = False
) -> None:
    """
    Set up logging format in a consistent way across projects

    Args:
        filename:
            Name of file to which logging messages will be written
        verbose:
            If true, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger()
    if filename is not None:
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(filename=filename, mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
