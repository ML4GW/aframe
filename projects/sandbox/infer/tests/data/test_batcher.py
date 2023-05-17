import numpy as np
import pytest
from infer.data.batcher import batch_chunks


@pytest.fixture(params=[1, 2, 5])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 10])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[128, 256])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[4, 10])
def chunk_size(request):
    return request.param


@pytest.fixture
def it(sample_rate, chunk_size):
    total_length = 100
    num_chunks = (total_length - 1) // chunk_size + 1

    def _it():
        for i in range(num_chunks):
            start = int(sample_rate * chunk_size * i)
            stop = start + int(sample_rate * chunk_size)
            stop = min(total_length * sample_rate, stop)

            x = np.arange(start, stop)
            x = np.stack([x, x + 1])
            yield x, -x

    return _it


def test_batch_chunks(batch_size, inference_sampling_rate, sample_rate, it):
    stride = int(sample_rate // inference_sampling_rate)
    step_size = int(batch_size * stride)
    num_steps = int(100 * sample_rate) // step_size
    chunker = batch_chunks(
        it(),
        num_steps,
        batch_size,
        inference_sampling_rate,
        sample_rate,
        throughput=10000,
    )
    xs, x_injs = [], []
    for i, (x, x_inj) in enumerate(chunker):
        assert x.shape == (2, step_size)
        xs.append(x)
        x_injs.append(x_inj)

    assert i == (num_steps - 1)
    x = np.concatenate(xs, axis=1)
    x_inj = np.concatenate(x_injs, axis=1)

    expected = np.arange(num_steps * step_size)
    expected = np.stack([expected, expected + 1])
    np.testing.assert_equal(x, expected)
    np.testing.assert_equal(x_inj, -expected)
