import logging
import sys
from logging.handlers import QueueHandler
from multiprocessing import Process, Queue
from typing import Union, Optional
import traceback
import subprocess


def subprocess_wrapper(
    f: callable,
):
    """
    Wraps a callable so that errors are propogated
    into a queue object, and logs are passed to a log_queue
    object for downstream handling by logger subprocess
    """

    def wrapper(
        error_queue: Queue,
        level: Union[int, str],
        name: str,
        log_queue: Optional[Queue],
        *args,
        **kwargs,
    ):
        # add a queue handler to the root logger;
        # in python logging, any child loggers
        # will by default inherit the parent logger handler
        if log_queue is not None:
            h = QueueHandler(log_queue)
            root = logging.getLogger()
            # clear already inherited handlers so
            # duplicate logs arent sent
            root.handlers.clear()
            root.addHandler(h)
            root.setLevel(level)

        try:
            f(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            error_queue.put((name, e, tb))

    return wrapper


def cleanup_subprocesses(
    subprocesses: list[Process],
):
    """
    Terminate and clean up subprocesses if the
    main process exits
    """

    for process in subprocesses:
        try:
            if process.is_alive():
                process.terminate()
                process.join()
        except Exception as e:
            # TODO: is there something better to do here?
            # If there's an exception in cleaning up one
            # subprocess, it shouldn't prevent the other
            # subprocsses from getting terminated, but
            # it still leaves something running.
            logging.info(f"Terminating process failed with exception {str(e)}")


def signal_handler(signum, frame):
    """
    Handle signals so that os signals can
    trigger subprocess cleanup
    """
    logging.debug(f"Received signal: {signum}")
    sys.exit(0)


def run_subprocess_with_logging(
    args: list[str], logger=None, log_stderr_on_success=False
):
    """
    Run a subprocess, logging stdout via a python logger

    If the subprocess succeeds (i.e return code 0),
    stdout is always logged via the logger.
    stderr can optionally be logged via the logger
    too for processes that write
    info to stderr on success,
    which is the default for python logging

    If the subprocess fails (i.e. non-zero return code)
    stderr is written directly to sys.stderr

    Args:
        args:
            Command line arguments for the subprocess
        logger:
            Logger to use (defaults to root logger if None)
        log_stderr_on_success:
            Whether to log stderr via logger when process succeeds

    Yields:
        The subprocess.CompletedProcess instance
    """
    if logger is None:
        logger = logging.getLogger()

    try:
        # run the subprocess and capture output
        result = subprocess.run(args, capture_output=True, text=True)

        # if process succeeds
        if result.returncode == 0:
            # always log stdout
            for line in result.stdout.splitlines():
                logger.info(line)

            # optionally log stderr too
            if log_stderr_on_success and result.stderr:
                for line in result.stderr.splitlines():
                    logger.info(line)
        # if process fails
        else:
            # write stderr directly to sys.stderr
            sys.stderr.write(result.stderr)
            sys.stderr.flush()

        # yield the result
        return result

    # handle errors related to logging
    # subprocess outputs
    except Exception as e:
        sys.stderr.write(f"Error running subprocess: {e}\n")
        sys.stderr.flush()
        raise
