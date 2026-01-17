import logging
from pathlib import Path


from utils.logging import configure_logging


def test_configure_logging_stdout_only():
    configure_logging(verbose=False)

    logger = logging.getLogger()
    assert logger.level == logging.INFO
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_configure_logging_verbose():
    configure_logging(verbose=True)

    logger = logging.getLogger()
    assert logger.level == logging.DEBUG


def test_configure_logging_with_file(tmp_path):
    log_file = tmp_path / "test.log"
    configure_logging(filename=log_file)

    logger = logging.getLogger()
    # Should have the StreamHandler and the FileHandler
    handlers = logger.handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]

    assert Path(file_handlers[0].baseFilename) == log_file.absolute()


def test_log_output_to_file(tmp_path):
    log_file = tmp_path / "output.log"
    configure_logging(filename=log_file, verbose=True)

    test_message = "Test log message"
    logging.getLogger("test_logger").debug(test_message)

    # Ensure file exists and contains the message
    assert log_file.exists()
    content = log_file.read_text()
    assert test_message in content
    assert "DEBUG" in content
