#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import numpy as np
import pytest

import bbhnet.injection
from bbhnet.injection import end_o3_ratesandpops, nonspin_bbh

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(params=[nonspin_bbh, end_o3_ratesandpops])
def prior(request):
    return request.param


@pytest.fixture(params=["IMRPhenomPv2"])
def approximant(request):
    return request.param


@pytest.fixture(params=["20", "50"])
def reference_frequency(request):
    return request.param


@pytest.fixture(params=[10, 20])
def minimum_frequency(request):
    return request.param


@pytest.fixture(params=[1, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def waveform_duration(request):
    return request.param


def test_generate_gw(
    prior,
    approximant,
    sample_rate,
    waveform_duration,
    minimum_frequency,
    reference_frequency,
    n_samples,
):

    n_pols = 2
    prior = prior()
    sample_params = prior.sample(n_samples)

    waveforms = bbhnet.injection.injection.generate_gw(
        sample_params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        approximant,
    )

    expected_waveform_size = waveform_duration * sample_rate
    expected_num_waveforms = n_samples

    assert waveforms.shape == (
        expected_num_waveforms,
        n_pols,
        expected_waveform_size,
    )


def test_inject_waveforms(sample_rate):
    times = np.arange(0, 10, 1 / sample_rate)
    background = np.zeros_like(times, dtype=np.float32)

    waveform_size = 4
    signal_times = np.arange(3, 10, 0.5)
    n_waveforms = len(signal_times)
    waveforms = np.ones((n_waveforms, waveform_size), dtype=np.float32)

    injected = bbhnet.injection.injection.inject_waveforms(
        (times, background), waveforms, signal_times
    )
    assert len(background) == len(injected)

    for i in range(len(background)):
        # account for offset from center of waveform to
        # get to first sample of waveform
        if (i + waveform_size // 2) / sample_rate < 3:
            # no signals in first 3 seconds
            assert injected[i] == 0, i
            continue

        # check if this sample is supposed to be in a waveform
        for j in range(waveform_size):
            # include check on divisor since the modulo would be zero
            # for a signal at 10s, but we don't have one there
            div, mod = divmod(i + waveform_size // 2 - j, sample_rate // 2)
            if mod == 0 and div < 20:
                assert injected[i] == 1, (i, j)
                break
        else:
            # otherwise make sure it's still 0
            assert injected[i] == 0, i
