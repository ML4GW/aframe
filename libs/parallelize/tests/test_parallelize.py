import time
from concurrent.futures import (
    CancelledError,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)

import pytest

from bbhnet.parallelize import AsyncExecutor, as_completed


def func(x):
    return x**2


def func_with_args(x, i):
    return x**i


def sleepy_func(t):
    time.sleep(t)
    return t


@pytest.fixture(params=[list, dict])
def container(request):
    values = range(10)
    if request.param is list:
        return list(values)
    else:
        letters = "abcdefghij"
        return {i: [j] for i, j in zip(letters, values)}


@pytest.fixture(params=[ThreadPoolExecutor, ProcessPoolExecutor])
def pool_type(request):
    return request.param


def test_as_completed(container, pool_type):
    futures = []
    with pool_type(2) as ex:
        if isinstance(container, dict):
            futures = {
                i: [ex.submit(func, k)]
                for i, j in container.items()
                for k in j
            }
        else:
            futures = [ex.submit(func, i) for i in container]

        for result in as_completed(futures):
            if isinstance(container, dict):
                letter, value = result
                letters = sorted(container.keys())
                assert value == letters.index(letter) ** 2
            else:
                assert 0 <= result**0.5 <= 9


def test_async_executor(pool_type):
    ex = AsyncExecutor(2, thread=pool_type is ThreadPoolExecutor)
    with pytest.raises(ValueError):
        ex.submit(func, 0)

    with ex:
        future = ex.submit(func, 2)
        assert future.result() == 4

        # test imap
        it = ex.imap(func, range(10))
        results = sorted([i for i in it])
        assert all([i == j**2 for i, j in zip(results, range(10))])

        it = ex.imap(func_with_args, range(10), i=3)
        results = sorted([i for i in it])
        assert all([i == j**3 for i, j in zip(results, range(10))])

        # submit more jobs than workers so that
        # the last will get started even if the
        # context exits as long as there's no error
        futures = [ex.submit(sleepy_func, 0.1) for i in range(3)]
        tstamp = time.time()

    # make sure that the context didn't move on until
    # everything finished executing
    assert (time.time() - tstamp) > 0.1

    # make sure all the jobs got executed
    assert all([f.result() == 0.1 for f in futures])

    # make sure that we have no more executor
    with pytest.raises(ValueError):
        ex.submit(func, 0)

    try:
        ex = AsyncExecutor(2, thread=pool_type is ThreadPoolExecutor)
        with ex:
            # pad with a couple extra to make sure that
            # one doesn't sneak in
            futures = [ex.submit(sleepy_func, 0.1) for i in range(5)]
            raise ValueError
    except ValueError:
        assert all([f.result() == 0.1 for f in futures[:2]])
        with pytest.raises(CancelledError):
            futures[-1].result()
