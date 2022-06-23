from unittest.mock import patch

import numpy as np
import pytest

from bbhnet.data import utils as data_utils


@pytest.fixture(params=[0, -4, -8, -20, -40, 4, 8, 20, 40])
def trigger_distance(request):
    return request.param


@pytest.fixture(params=[16, 20, 64])
def size(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


@pytest.fixture(params=[None, 4, 8, 11])
def N(request):
    return request.param


@pytest.fixture(params=[8, 40, 100])
def trigger_distance_size(request):
    return request.param


def test_sample_kernels(ndim, size, N):
    # set trigger_dist_size = size / 2
    # this replicates behavior
    # that allows t0 to lie anywhere in kernel

    trigger_dist_size = 0

    # for 1D arrays we need more data so that we
    # have enough to sample across
    if ndim == 1 and N is None:
        xsize = size
    elif ndim == 1:
        xsize = N * size * 2
    else:
        xsize = 100

    # set up a dummy array for sampling from
    x = np.arange(xsize)
    if ndim > 1:
        x = np.stack([x + i * xsize for i in range(8)])
        if ndim == 3:
            x = x[:, None]
            x = np.concatenate([x, x + xsize], axis=1)

    if ndim == 1 and N is None:
        # must specify number of samples for ndim == 1
        with pytest.raises(ValueError):
            data_utils.sample_kernels(x, size, trigger_dist_size, N)
        return
    elif ndim == 1:
        # timeseries has to be long enough to sample
        # N kernels of size size
        with pytest.raises(ValueError):
            data_utils.sample_kernels(
                x[: N + size - 1], size, trigger_dist_size, N
            )

    # make sure we returned the appropriate number of kernels
    kernels = data_utils.sample_kernels(x, size, trigger_dist_size, N)
    if N is not None:
        assert len(kernels) == N
    else:
        assert len(kernels) == 8

    # make sure that the kernels all have the expected shape
    expected_shape = (size,)
    if ndim == 3:
        expected_shape = (2,) + expected_shape
    assert all([i.shape == expected_shape for i in kernels])

    # verify kernel content
    if ndim == 1:
        # 1D case is easy
        for kernel in kernels:
            i = kernel[0]
            assert (kernel == np.arange(i, i + size)).all()
    elif ndim == 2:
        # 2D needs to check more
        idx_seen = []
        for i, kernel in enumerate(kernels):
            # make sure the center of the timeseries is in kernel
            assert xsize // 2 in kernel % xsize

            # make sure that the kernel is all contiguous ints
            j = kernel[0]
            assert (kernel == np.arange(j, j + size)).all()

            # keep track of which samples the kernels were
            # sampled from to make sure that there's no
            # overlap if N < len(x)
            if N is not None:
                idx_seen.append(j // xsize)
            else:
                assert j // xsize == i

        # verify that there's no overlapping samples
        if N is not None and N <= len(x):
            assert len(idx_seen) == len(list(set(idx_seen)))
    else:
        # similar tests for 3D case, but need to make
        # sure that we have the same slice from each channel
        idx_seen = []
        for i, kernel in enumerate(kernels):
            # verify center of timeseries in kernel
            assert xsize // 2 in kernel[0] % xsize

            # verify contiguous ints and that we have
            # the same slice in each channel
            j = kernel[0, 0]
            expected = np.arange(j, j + size)
            expected = np.stack([expected, expected + xsize])
            assert (kernel == expected).all()

            # keep track of which samples kernels are from
            if N is not None:
                idx_seen.append(j // xsize)
            else:
                assert j // xsize == i

        # verify no overlapping samples
        if N is not None and N <= len(x):
            assert len(idx_seen) == len(list(set(idx_seen)))


def test_sample_kernels_with_trigger_distance(size, trigger_distance_size, N):

    # create dummy arrays
    # where the difference between
    # two values in the array
    # is also the amount of samples apart

    xsize = 200
    x = np.arange(xsize)
    x = np.stack([x + i * xsize for i in range(8)])

    t0_value = xsize // 2

    if trigger_distance_size < 0 and abs(trigger_distance_size) >= (size / 2):
        with pytest.raises(ValueError):
            data_utils.sample_kernels(x, size, trigger_distance_size, N)
        return

    else:

        # patch so that either the max or min sample is returned
        with patch(
            "bbhnet.data.utils.np.random.randint",
            new=lambda i, j: np.random.choice([i, j]),
        ):
            kernels = data_utils.sample_kernels(
                x, size, trigger_distance_size, N
            )

        for kernel in kernels:
            diffs = (kernel % xsize) - t0_value
            closest_edge = np.abs([diffs.min(), diffs.max()]).min()

            if trigger_distance_size < 0:
                # for negative trigger distances
                # t0 should always be in kernel
                assert t0_value in (kernel % xsize)

                # the closest edge should be greater than trigger
                # distance size away
                assert closest_edge >= abs(trigger_distance_size)

            elif trigger_distance_size >= 0:

                # for positive trigger distance sizes
                # either the sample is in the kernel,
                # or a maximum of trigger_distance away
                # from closest edge
                assert closest_edge <= trigger_distance_size
