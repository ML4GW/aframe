from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed as _as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Union

FutureList = Iterable[Future]


def _handle_future(future: Future):
    exc = future.exception()
    if exc is not None:
        raise exc
    return future.result()


def _lazy_as_completed(futures: Iterable[Future]):
    while len(futures) > 0:
        f = futures.pop(0)
        if f.done():
            yield _handle_future(f)
        else:
            futures.append(f)


def as_completed(futures: Union[FutureList, Dict[Any, FutureList]]):
    """
    Extension of `concurrent.futures.as_completed` that supports
    braided iteration through dictionaries of iterables of futures,
    yielding for each element in those iterables both the key of
    the dictionary element containing the future's iterable and
    the result of the futures itself.
    """
    if isinstance(futures, dict):
        futures = {k: _lazy_as_completed(v) for k, v in futures.items()}
        while len(futures) > 0:
            keys = list(futures.keys())
            for key in keys:
                try:
                    result = next(futures[key])
                except StopIteration:
                    futures.pop(key)
                    continue
                else:
                    yield key, result
    else:
        for future in _as_completed(futures):
            yield _handle_future(future)


@dataclass
class AsyncExecutor:
    workers: int
    thread: bool = True

    def __post_init__(self):
        self._executor = None

    def __enter__(self):
        if self.thread:
            self._executor = ThreadPoolExecutor(self.workers)
        else:
            self._executor = ProcessPoolExecutor(self.workers)
        return self

    def __exit__(self, *exc_args):
        # cancel futures if we hit an error, since we're
        # going to assume that this means something was wrong
        # with the future function that got called
        cancel_futures = exc_args[0] is not None
        self._executor.shutdown(wait=True, cancel_futures=cancel_futures)
        self._executor = None

    def submit(self, *args, **kwargs):
        if self._executor is None:
            raise ValueError("AsyncExecutor has no executor to submit jobs to")
        return self._executor.submit(*args, **kwargs)

    def imap(self, f: Callable, it: Iterable, **kwargs: Any):
        if self._executor is None:
            raise ValueError("AsyncExecutor has no executor to submit jobs to")

        futures = [self.submit(f, i, **kwargs) for i in it]
        return as_completed(futures)
