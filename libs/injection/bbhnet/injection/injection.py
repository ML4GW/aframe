#!/usr/bin/env python
# coding: utf-8

import logging
import os
import sys

import bilby
import h5py
import numpy as np
import scipy.signal as sig
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from gwpy.timeseries import TimeSeries

logging.basicConfig(
    format="%(asctime)s - %(message)s", level=logging.INFO, stream=sys.stdout
)


def get_snr(data, noise_psd, fs, fmin=20):
    """ Calculate the waveform SNR given the background noise PSD"""

    data_fd = np.fft.rfft(data) / fs
    data_freq = np.fft.rfftfreq(len(data)) * fs
    dfreq = data_freq[1] - data_freq[0]

    noise_psd_interp = noise_psd.interpolate(dfreq)
    noise_psd_interp[noise_psd_interp == 0] = 1.0

    snr = 4 * np.abs(data_fd) ** 2 / noise_psd_interp.value * dfreq
    snr = np.sum(snr[fmin <= data_freq])
    snr = np.sqrt(snr)

    return snr


def generate_gw(
    sample_params,
    ifo,
    waveform_generator=None,
    get_snr=False,
    noise_psd=None,
    whiten_fn=None,
):
    """Generate gravitational-wave events

    Arguments:
    - sample_params: dictionary of GW parameters
    - sample_duration: time duration of each sample
    - triggers: trigger time (relative to `sample_duration`) of each sample
    - ifo: interferometer
    - waveform_generator: bilby.gw.WaveformGenerator with appropriate params
    - get_snr: return the SNR of each sample
    - noise_psd: background noise PSD used to calculate SNR the sample
    - whiten_fn: whiten each sample using the background noise PSD
    """

    n_sample = len(sample_params["geocent_time"])

    if waveform_generator is None:
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=8,
            sampling_frequency=16384,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomPv2",
                "reference_frequency": 50,
                "minimum_frequency": 20,
            },
        )

    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration
    waveform_size = int(sample_rate * waveform_duration)

    signals = np.zeros((n_sample, waveform_size))
    snr = np.zeros(n_sample)

    ifo = bilby.gw.detector.get_empty_interferometer(ifo)
    for i in range(n_sample):
        # Get parameter for one signal
        p = dict()
        for k, v in sample_params.items():
            p[k] = v[i]
        ra, dec, geocent_time, psi = (
            p["ra"],
            p["dec"],
            p["geocent_time"],
            p["psi"],
        )

        # Generate signal in IFO
        polarizations = waveform_generator.time_domain_strain(p)
        signal = np.zeros(waveform_size)
        b, a = sig.butter(
            N=8,
            Wn=waveform_generator.waveform_arguments["minimum_frequency"],
            btype="highpass",
            fs=waveform_generator.sampling_frequency,
        )
        for mode in polarizations.keys():
            # Get ifo response
            response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
            polarizations[mode] = sig.filtfilt(b, a, polarizations[mode])
            signal += polarizations[mode] * response

        # Total shift = shift to trigger time + geometric shift
        dt = waveform_duration / 2.0
        dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)
        signal = np.roll(signal, int(np.round(dt * sample_rate)))

        # Calculate SNR
        if noise_psd is not None:
            if get_snr:
                snr[i] = get_snr(signal, noise_psd, sample_rate)

        signals[i] = signal
    if get_snr:
        return signals, snr
    return signals


def inject_signals(
    frame_files: [str],
    channels: [str],
    ifos: [str],
    prior_file: str,
    n_samples: int,
    outdir: str,
    fmin: float = 20,
    waveform_duration: float = 8,
    snr_range: [float] = [25, 50],
):

    """ Start simulation """

    strain = TimeSeries.read(frame_files[0], channels[0])
    logging.info(f"Read strain from {frame_files[0]}")

    frame_start, frame_stop = strain.span
    frame_duration = frame_stop - frame_start

    sample_rate = int(strain.sample_rate.value)
    fftlength = int(max(2, np.ceil(2048 / sample_rate)))

    # set the non-overlapping times of the signals in the frames randomly
    # leaves buffer at either end of the series so edge effects aren't an issue
    signal_times = sorted(
        np.random.choice(
            np.arange(
                waveform_duration,
                frame_duration - waveform_duration,
                waveform_duration,
            ),
            size=n_samples,
            replace=False,
        )
    )

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(n_samples))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
    logging.info("High pass filter [Hz] : {}".format(fmin))
    logging.info("Prior file            : {}".format(prior_file))

    # define a Bilby waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
            "minimum_frequency": 20,
        },
    )

    # sample GW parameters from prior distribution
    priors = bilby.gw.prior.BBHPriorDict(prior_file)
    sample_params = priors.sample(n_samples)

    # Come back to this and see if can add frame_start
    sample_params["geocent_time"] = signal_times

    signal_strains = []
    signals_list = []
    snr_list = []
    for frame, channel, ifo in zip(frame_files, channels, ifos):
        # Get the strain from each frame file
        strain = TimeSeries.read(frame, channel)

        # calculate the PSD
        strain_psd = strain.psd(fftlength)

        # generate GW waveforms
        signals, snr = generate_gw(
            sample_params,
            ifo,
            waveform_generator=waveform_generator,
            get_snr=True,
            noise_psd=strain_psd,
        )

        signal_strains.append(strain)
        signals_list.append(signals)
        snr_list.append(snr)

    old_snr = np.sqrt(np.sum(np.square(snr_list)))
    new_snr = np.random.uniform(snr_range[0], snr_range[1], len(snr_list[0]))

    signals_list = [
        signals * (new_snr / old_snr)[:, None] for signals in signals_list
    ]
    sample_params["luminosity_distance"] = (
        sample_params["luminosity_distance"] * old_snr / new_snr
    )
    snr_list = [snr * new_snr / old_snr for snr in snr_list]

    for strain, signals, frame in zip(
        signal_strains, signals_list, frame_files
    ):
        for i in range(n_samples):
            idx1 = int(
                (signal_times[i] - waveform_duration / 2.0) * sample_rate
            )
            idx2 = idx1 + waveform_duration * sample_rate
            strain[idx1:idx2] += signals[i]

        strain.write(os.path.join(outdir, frame))

    # Write params and similar to output file
    param_file = os.path.join(
        outdir, f"param_file_{frame_start}-{frame_stop}.h5"
    )
    with h5py.File(param_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        params_gr = f.create_group("signal_params")
        for k, v in sample_params.items():
            params_gr.create_dataset(k, data=v)

        # Save signal times as actual GPS times
        f.create_dataset("GPS-start", data=signal_times + frame_start)

        for i, ifo in enumerate(ifos):
            ifo_gr = f.create_group(ifo)
            ifo_gr.create_dataset("signal", data=signals_list[i])
            ifo_gr.create_dataset("snr", data=snr_list[i])

        # write frame attributes
        f.attrs.update(
            {
                "size": n_samples,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "sample_rate": sample_rate,
                "psd_fftlength": fftlength,
            }
        )

        # Update signal attributes
        f.attrs["waveform_duration"] = waveform_duration
        f.attrs["flag"] = "GW"
