from multiprocessing import Queue


def subprocess_wrapper(
    f: callable,
):
    """
    Wraps a callable so that errors are propogated
    into a queue object
    """

    def wrapper(error_queue: Queue, *args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            error_queue.put(e)
        return

    return wrapper
