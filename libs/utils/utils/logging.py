import logging
import sys
from pathlib import Path


def configure_logging(
    filename: str | Path | None = None, verbose: bool = False
) -> None:
    """
    Configure logging format consistently across projects.

    Sets up a logger with a standard format that includes timestamp,
    logger name, log level, and message. Optionally writes logs to
    both stdout and a file.

    Args:
        filename (str or Path, optional): Path to file where logging messages
            will be written. If None, logs are written only to stdout.
            Defaults to None.
        verbose (bool, optional): If True, log at DEBUG verbosity level.
            If False, log at INFO level. Defaults to False.

    Returns:
        None

    Examples:
        >>> configure_logging(verbose=True)  # Log at DEBUG level to stdout
        >>> configure_logging('logs.txt')    # INFO to file
        >>> configure_logging('debug.log', verbose=True)  # DEBUG to file
    """
    # Define log format with timestamp, logger name, level, and message
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger with appropriate verbosity level
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
        force=True,
    )

    # If a filename is provided, add file handler to write logs to file
    logger = logging.getLogger()
    if filename is not None:
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(filename=filename, mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
