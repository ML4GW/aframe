from unittest.mock import MagicMock, patch

import h5py
import pytest
import torch
from datagen.scripts.timeslide_waveforms import main
from datagen.utils.timeslide_waveforms import calc_segment_injection_times

from aframe.priors.priors import end_o3_ratesandpops


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


def create_mock_compute_network_snr(snr_threshold):
    def f(projected, *args):
        return torch.randn(len(projected)) + snr_threshold

    return f


@pytest.fixture()
def mock_psds(sample_rate, waveform_duration):
    n_psd_samples = int((sample_rate / 2) / (1 / waveform_duration)) + 1
    mock_psds = torch.randn(2, n_psd_samples)
    return mock_psds


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
            background_dir=background_mock,
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


def test_main_for_seeding(prior, waveform_approximant, tmp_path):
    background_mock = MagicMock()
    background_mock.iterdir = lambda: iter([None])
    mock_compute_network_snr = create_mock_compute_network_snr(4)
    ml4gw_mock = patch(
        "datagen.scripts.timeslide_waveforms.compute_network_snr",
        new=mock_compute_network_snr,
    )
    load_mock = patch(
        "datagen.utils.timeslide_waveforms.load_psds",
        return_value=mock_psds,
    )

    def run_main(shift, seed, i):
        torch.manual_seed(seed)
        with load_mock, ml4gw_mock:
            return main(
                0,
                100,
                shifts=[0, shift],
                background_dir=background_mock,
                spacing=6,
                buffer=4,
                waveform_duration=3,
                prior=prior,
                minimum_frequency=20,
                reference_frequency=20,
                sample_rate=256,
                waveform_approximant=waveform_approximant,
                highpass=32,
                snr_threshold=4,
                ifos=["H1", "L1"],
                output_dir=tmp_path / f"test-{i}.h5",
                seed=seed,
            )

    def compare(x1, x2, key, equal):
        x1, x2 = x1[key][:], x2[key][:]
        if equal:
            assert (x1 == x2).all(), key
        else:
            assert (x1 != x2).all(), key

    def verify_datasets(results1, results2, equal):
        output_fname1, rejected_params1 = results1
        output_fname2, rejected_params2 = results2

        f1 = h5py.File(output_fname1)
        f2 = h5py.File(output_fname2)
        with f1, f2:
            for dataset in ["parameters", "waveforms"]:
                for key in f1[dataset].keys():
                    if key in ["gps_time", "shift", "snr"] and not equal:
                        # timestamps should be equal in all cases
                        # 0 values of shifts will be same
                        # we're faking SNRs so those will be same too
                        continue
                    compare(f1[dataset], f2[dataset], key, equal)

        f1 = h5py.File(rejected_params1)
        f2 = h5py.File(rejected_params2)
        with f1, f2:
            dataset = "parameters"
            for key in f1[dataset].keys():
                if key == "snr" and not equal:
                    continue
                compare(f1[dataset], f2[dataset], key, equal)

    # check that running with the same seed
    # generates the same results
    results1 = run_main(1, 42, 1)
    results2 = run_main(1, 42, 2)
    verify_datasets(results1, results2, True)

    # now check that running with the same seed but
    # using different shift generates different results
    results3 = run_main(2, 42, 3)
    verify_datasets(results1, results3, False)

    # finally verify that using this shift/seed
    # combination again once again generates
    # identical results
    results4 = run_main(2, 42, 4)
    verify_datasets(results3, results4, True)
