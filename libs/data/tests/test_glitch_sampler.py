from unittest.mock import patch

import numpy as np

from bbhnet.data.glitch_sampler import GlitchSampler


def test_glitch_sampler(
    arange_glitches, glitch_length, sample_rate, data_length, device
):
    sampler = GlitchSampler(arange_glitches)
    sampler.to(device)
    assert sampler.hanford.device.type == device
    assert sampler.livingston.device.type == device

    assert sampler.hanford.shape == (10, glitch_length * sample_rate)
    assert sampler.livingston.shape == (10, glitch_length * sample_rate)

    # take over randint to return 4 so that we
    # know what size arrays to expect and verify
    # that they come out correctly
    with patch("numpy.random.randint", return_value=4):
        hanford, livingston = sampler.sample(8, data_length)
    assert hanford.shape == (4, data_length)
    assert livingston.shape == (4, data_length)

    # resample these since setting randint to equal
    # 4 actually throws off sample_kernels when it
    # gets called under the hood. Use this cheap little
    # loop to make sure that we have some data for both
    # interferometers to verify
    hanford = livingston = None
    while hanford is None or livingston is None:
        hanford, livingston = sampler.sample(8, data_length)

    glitch_size = glitch_length * sample_rate
    for i, tensor in enumerate([hanford, livingston]):
        value = tensor.cpu().numpy()

        # the livingston data is negative
        power = (-1) ** i

        # make sure each sampled glitch matches our expectations
        for row in value:
            # make sure that the "trigger" of each glitch aka
            # the center value is in each sample
            assert glitch_size // 2 in row % glitch_size

            # make sure that the sampled glitch is a
            # contiguous chunk of ints
            j = row[0]
            assert (row == np.arange(j, j + power * data_length, power)).all()

    # now make sure that Nones get returned when
    # all the glitches are one or the other
    with patch("numpy.random.randint", return_value=0):
        hanford, livingston = sampler.sample(8, data_length)
    assert hanford is None
    assert livingston.shape == (8, data_length)

    with patch("numpy.random.randint", return_value=8):
        hanford, livingston = sampler.sample(8, data_length)
    assert livingston is None
    assert hanford.shape == (8, data_length)
