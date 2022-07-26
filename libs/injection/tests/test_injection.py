#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.detector import Channel
from gwpy.timeseries import TimeSeries

import bbhnet.injection
from bbhnet.parallelize import AsyncExecutor

TEST_DIR = Path(__file__).resolve().parent


def mock_frame(sample_rate, duration):
    mock_data = np.random.random(int(sample_rate * duration))
    return TimeSeries(
        mock_data,
        unit=None,
        t0=0,
        sample_rate=sample_rate,
        channel=Channel("IFO:Strain", sample_rate=sample_rate, dtype="float"),
        name="Strain",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        (dict(duration=16),),
        (dict(sampling_frequency=128, waveform_approximant="TaylorF2"),),
    ],
)
def test_get_waveform_generator(kwargs):
    (kwargs,) = kwargs
    waveform_generator = bbhnet.injection.injection.get_waveform_generator(
        **kwargs
    )
    sampling_kwargs = {
        "duration",
        "sampling_frequency",
        "frequency_domain_source_model",
        "parameter_conversion",
    }
    waveform_kwargs = {
        "waveform_approximant",
        "reference_frequency",
        "minimum_frequency",
    }

    for k in sampling_kwargs:
        assert hasattr(waveform_generator, k)
        if k in kwargs:
            assert getattr(waveform_generator, k) == kwargs[k]

    assert hasattr(waveform_generator, "waveform_arguments")

    for k in waveform_kwargs:
        assert waveform_generator.waveform_arguments.get(k)
        if k in kwargs:
            assert waveform_generator.waveform_arguments.get(k) == kwargs[k]


@patch("bbhnet.injection.injection.apply_high_pass_filter")
def test_generate_gw(mock_filter):
    """Test generate_gw using supplied waveform generator, or
    initializing generator
    """
    import bilby

    sample_params = bilby.gw.prior.BBHPriorDict().sample(10)

    with patch("bilby.gw.waveform_generator.WaveformGenerator") as mock_gen:
        bbhnet.injection.injection.generate_gw(
            sample_params,
            waveform_generator=None,
            waveform_approximant="TaylorF2",
            sampling_frequency=128,
        )

    assert mock_gen.call_count == 1

    dummy_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        duration=1,
        sampling_frequency=128,
    )
    mock_gen.reset_mock()
    with patch("bilby.gw.waveform_generator.WaveformGenerator") as mock_gen:
        bbhnet.injection.injection.generate_gw(
            sample_params, waveform_generator=dummy_waveform_generator
        )

    assert mock_gen.call_count == 0


@pytest.mark.parametrize(
    "ifos,sample_rate,spacing,file_length,fmin,prior",
    [
        (["H1", "L1"], 2048, 60, 4096, 32, "nonspin_BBH.prior"),
    ],
)
def test_inject_signals_into_timeslide(
    raw_timeslide,
    inj_timeslide,
    ifos,
    sample_rate,
    spacing,
    file_length,
    fmin,
    prior,
):

    # create process and thread pools
    thread_ex = AsyncExecutor(4, thread=True)
    process_ex = AsyncExecutor(4, thread=False)

    # TODO: why did we get rid of this fixture? just re using code
    prior_file = Path(__file__).absolute().parent / "prior_files"
    prior_file /= prior
    prior_file = prior_file.as_posix()

    with process_ex, thread_ex:
        out_timeslide = bbhnet.injection.inject_signals_into_timeslide(
            process_ex,
            thread_ex,
            raw_timeslide,
            inj_timeslide,
            ifos,
            prior_file,
            spacing,
            sample_rate,
            file_length,
            fmin,
        )

    param_file = out_timeslide.path / "params.h5"
    assert out_timeslide.path.exists()
    assert param_file.exists()
