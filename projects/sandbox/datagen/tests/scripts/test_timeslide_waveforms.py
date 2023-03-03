from unittest.mock import patch

import h5py
import pytest
import torch
from astropy.cosmology import Planck15
from datagen.scripts.timeslide_waveforms import main
from datagen.utils.timeslide_waveforms import (
    calc_segment_injection_times,
    calc_shifts_required,
)

from bbhnet.priors.priors import end_o3_ratesandpops


@pytest.fixture(params=[end_o3_ratesandpops])
def prior(request):
    return request.param


@pytest.fixture(params=[30, 60])
def spacing(request):
    return request.param


@pytest.fixture(params=[2])
def buffer(request):
    return request.param


@pytest.fixture(params=[8])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[32])
def highpass(request):
    return request.param


@pytest.fixture(params=[20])
def minimum_frequency(request):
    return request.param


@pytest.fixture(params=[50])
def reference_frequency(request):
    return request.param


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[8])
def snr_threshold(request):
    return request.param


@pytest.fixture(params=[[0, 1]])
def shifts(request):
    return request.param


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=["HOFT_C01"])
def frame_type(request):
    return request.param


@pytest.fixture(params=["IMRPhenomPv2"])
def waveform_approximant(request):
    return request.param


@pytest.fixture(params=["DCS-ANALYSIS_READY_C01:1"])
def state_flag(request):
    return request.param


@pytest.fixture()
def cosmology():
    def f():
        return Planck15

    return f


def create_mock_compute_network_snr(snr_threshold):
    def f(projected, *args):
        return torch.randn(len(projected)) + snr_threshold

    return f


@pytest.fixture()
def mock_psds(sample_rate, waveform_duration):
    n_psd_samples = int((sample_rate / 2) / (1 / waveform_duration)) + 1
    mock_psds = torch.randn(2, n_psd_samples)
    return mock_psds


def test_calc_shifts_required():
    # one shift has 30 seconds of livetime
    segments = ((0, 10), (20, 30), (40, 50))

    # test that requiring 0 background time returns 0 shifts
    shifts_required = calc_shifts_required(segments, 0, 1)
    assert shifts_required == 0

    # need an extra shift to get 60 seconds of background
    # due to the chopping off of livetime at the end of each segment
    shifts_required = calc_shifts_required(segments, 60, 1)
    assert shifts_required == 3


def test_main(
    spacing,
    buffer,
    waveform_duration,
    cosmology,
    prior,
    minimum_frequency,
    reference_frequency,
    sample_rate,
    waveform_approximant,
    highpass,
    snr_threshold,
    ifos,
    mock_psds,
    tmp_path,
):

    start, stop = 0, 1000
    hanford_background = livingston_background = None
    mock_compute_network_snr = create_mock_compute_network_snr(snr_threshold)
    with patch(
        "datagen.scripts.timeslide_waveforms.utils.load_psds",
        return_value=mock_psds,
    ), patch(
        "datagen.scripts.timeslide_waveforms.compute_network_snr",
        new=mock_compute_network_snr,
    ):
        output_fname = main(
            start,
            stop,
            hanford_background,
            livingston_background,
            spacing,
            buffer,
            waveform_duration,
            cosmology,
            prior,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_approximant,
            highpass,
            snr_threshold,
            ifos,
            tmp_path / "test.h5",
        )

        expected_n_signals = len(
            calc_segment_injection_times(
                start, stop, spacing, buffer, waveform_duration
            )
        )
        assert output_fname.exists()
        with h5py.File(output_fname, "r") as f:
            shape = f["signals"].shape
            assert shape == (
                expected_n_signals,
                2,
                waveform_duration * sample_rate,
            )

            for key in f.keys():
                data = f[key][:]
                assert len(data) == shape[0]
                if key == "snr":
                    assert all(data > snr_threshold)


def test_deploy():
    pass
