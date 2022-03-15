#!/usr/bin/env python
# coding: utf-8
import h5py
from generate_waveforms import main

prior_files = "/home/william.benoit/simulateWaveforms/prior_files/"
outdir = "/home/william.benoit/simulateWaveforms/"


def check_file_contents(
    signal_file, n_samples, waveform_duration, sample_rate
):

    signal_length = waveform_duration * sample_rate

    with h5py.File(signal_file, "r") as f:
        for key in f.keys():
            if key == "signals":
                act_shape = f[key].shape
                exp_shape = (n_samples, 2, signal_length)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for signals, found {act_shape}"
            else:
                act_shape = f[key].shape
                exp_shape = (n_samples,)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for {key}, found {act_shape}"


def basic_test():
    prior_file = prior_files + "nonspin_BBH.prior"
    n_samples = 100
    waveform_duration = 8
    sample_rate = 4096

    signal_file = main(
        prior_file,
        n_samples,
        outdir,
        waveform_duration=waveform_duration,
        sample_rate=sample_rate,
    )

    check_file_contents(signal_file, n_samples, waveform_duration, sample_rate)


def different_prior():
    prior_file = prior_files + "precess_tides.prior"
    n_samples = 100
    waveform_duration = 8
    sample_rate = 4096

    signal_file = main(
        prior_file,
        n_samples,
        outdir,
        waveform_duration=waveform_duration,
        sample_rate=sample_rate,
    )

    check_file_contents(signal_file, n_samples, waveform_duration, sample_rate)


def different_duration():
    prior_file = prior_files + "nonspin_BBH.prior"
    n_samples = 100
    waveform_duration = 4
    sample_rate = 4096

    signal_file = main(
        prior_file,
        n_samples,
        outdir,
        waveform_duration=waveform_duration,
        sample_rate=sample_rate,
    )

    check_file_contents(signal_file, n_samples, waveform_duration, sample_rate)


def different_sample_rate():
    prior_file = prior_files + "nonspin_BBH.prior"
    n_samples = 100
    waveform_duration = 8
    sample_rate = 16384

    signal_file = main(
        prior_file,
        n_samples,
        outdir,
        waveform_duration=waveform_duration,
        sample_rate=sample_rate,
    )

    check_file_contents(signal_file, n_samples, waveform_duration, sample_rate)


def different_n_samples():
    prior_file = prior_files + "nonspin_BBH.prior"
    n_samples = 10
    waveform_duration = 8
    sample_rate = 4096

    signal_file = main(
        prior_file,
        n_samples,
        outdir,
        waveform_duration=waveform_duration,
        sample_rate=sample_rate,
    )

    check_file_contents(signal_file, n_samples, waveform_duration, sample_rate)


if __name__ == "__main__":
    basic_test()
    different_prior()
    different_duration()
    different_sample_rate()
    different_n_samples()
