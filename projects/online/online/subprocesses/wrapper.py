from multiprocessing import Queue
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
            error_queue.put((name, str(e), tb))

    return wrapper
