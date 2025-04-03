import logging
import sys
from multiprocessing import Process, Queue
from typing import List
import traceback


def subprocess_wrapper(
    f: callable,
):
    """
    Wraps a callable so that errors are propogated
    into a queue object
    """

    def wrapper(error_queue: Queue, name: str, *args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            error_queue.put((name, e, tb))

    return wrapper


def cleanup_subprocesses(
    subprocesses: List[Process],
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
