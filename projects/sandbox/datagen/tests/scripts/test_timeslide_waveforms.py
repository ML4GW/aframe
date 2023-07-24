from unittest.mock import MagicMock, patch

import h5py
import pytest
import torch
from datagen.scripts.timeslide_waveforms import main
from datagen.utils.timeslide_waveforms import (
    calc_segment_injection_times,
    calc_shifts_required,
)

from aframe.priors.priors import mdc_prior_chirp_distance


@pytest.fixture(params=[mdc_prior_chirp_distance])
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

    # test that requiring 0 background time returns 0 shifts
    shifts_required = calc_shifts_required(0, 30, 1)
    assert shifts_required == 0

    # need an extra shift to get 60 seconds of background
    # due to the chopping off of livetime at the end of each segment
    shifts_required = calc_shifts_required(60, 30, 1)
    assert shifts_required == 3


def test_main(
    spacing,
    buffer,
    waveform_duration,
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
    background_mock = MagicMock()
    background_mock.iterdir = lambda: iter([None])
    mock_compute_network_snr = create_mock_compute_network_snr(snr_threshold)
    ml4gw_mock = patch(
        "datagen.scripts.timeslide_waveforms.compute_network_snr",
        new=mock_compute_network_snr,
    )
    load_mock = patch(
        "datagen.utils.timeslide_waveforms.load_psds",
        return_value=mock_psds,
    )
    with load_mock, ml4gw_mock:
        output_fname, rejected_params = main(
            start,
            stop,
            shifts=[0, 1],
            background=background_mock,
            spacing=spacing,
            buffer=buffer,
            waveform_duration=waveform_duration,
            prior=prior,
            minimum_frequency=minimum_frequency,
            reference_frequency=reference_frequency,
            sample_rate=sample_rate,
            waveform_approximant=waveform_approximant,
            highpass=highpass,
            snr_threshold=snr_threshold,
            ifos=ifos,
            output_dir=tmp_path / "test.h5",
        )

        expected_n_signals = len(
            calc_segment_injection_times(
                start, stop, spacing, buffer, waveform_duration
            )
        )
        assert output_fname.exists()

        # TODO: should really just load in using
        # the LigoResponseSetObject but this will
        # work for the time being
        with h5py.File(output_fname, "r") as f:
            num_injections = f.attrs["num_injections"]
            waveforms = f["waveforms"]
            for ifo in "hl":
                assert f"{ifo}1" in waveforms
            shape = waveforms["h1"].shape
            assert shape == (
                expected_n_signals,
                waveform_duration * sample_rate,
            )

            params = f["parameters"]
            for key in params:
                data = params[key][:]
                assert len(data) == shape[0]
                if key == "snr":
                    assert all(data > snr_threshold)

        with h5py.File(rejected_params, "r") as f:
            num_rejected = f.attrs["length"]
            assert (num_rejected + expected_n_signals) == num_injections

            params = f["parameters"]
            for key in params:
                data = params[key][:]
                assert len(data) == num_rejected
                if key == "snr":
                    assert all(data < snr_threshold)
