import numpy as np
import pytest
from infer.data.injector import Injector

from bbhnet.analysis.ledger.injections import LigoResponseSet


def ranger(sample_rate, N, step):
    step = int(sample_rate * step)
    for i in range(N):
        x = step * i + np.arange(step)
        yield np.stack([x, -x]).astype("float32")


@pytest.fixture(params=[128])
def sample_rate(request):
    return request.param


@pytest.fixture
def gps_times():
    return np.array([2.5, 5, 13, 17.25])


@pytest.fixture
def response_set(sample_rate, gps_times):
    duration = 1
    size = int(duration * sample_rate)
    N = len(gps_times)
    kwargs = {}

    fields = LigoResponseSet.__dataclass_fields__
    for name, attr in fields.items():
        if attr.metadata["kind"] == "parameter":
            kwargs[name] = np.zeros((N,))
        elif attr.metadata["kind"] == "waveform":
            wave = np.ones((N, size)) * (1 + np.arange(N)[:, None])
            kwargs[name] = wave.astype(np.float32)

    kwargs["gps_time"] = gps_times
    kwargs["num_injections"] = N
    kwargs["sample_rate"] = sample_rate
    kwargs["duration"] = duration
    return LigoResponseSet(**kwargs)


def test_injector(response_set, sample_rate, gps_times):
    it = ranger(sample_rate, 4, 5)

    injector = Injector(response_set, 0, sample_rate)
    it = map(injector, it)
    xs, x_injs = [], []
    for x, x_inj in it:
        assert x.shape == x_inj.shape
        xs.append(x)
        x_injs.append(x_inj)

    x = np.concatenate(xs, axis=1)
    x_inj = np.concatenate(x_injs, axis=1)
    expected = np.arange(sample_rate * 20)
    expected = np.stack([expected, -expected])
    assert (x == expected).all()
    assert not (x_inj == expected).all()

    idx = [0]
    for t in gps_times:
        for i in range(2):
            sign = (-1) ** (i + 1)
            tstamp = t + sign * 0.5
            idx.append(int(tstamp * sample_rate))

    for i, (start, stop) in enumerate(zip(idx[:-1], idx[1:])):
        expected = np.arange(start, stop)
        expected = np.stack([expected, -expected])
        div, mod = divmod(i, 2)
        if mod:
            expected += np.ones_like(expected) * (div + 1)
        np.testing.assert_equal(x_inj[:, start:stop], expected)
