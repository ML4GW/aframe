import logging

import bilby
import numpy as np
import scipy.signal as sig
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole


def calc_snr(data, noise_psd, fs, fmin=20):
    """Calculate the waveform SNR given the background noise PSD

    Args:
        data: timeseries of the signal whose SNR is to be calculated
        noise_psd: PSD of the background that the signal is in
        fs: sampling frequency of the signal and background
        fmin: minimum frequency for the highpass filter

    Returns:
        The SNR of the signal, a single value

    """

    data_fd = np.fft.rfft(data) / fs
    data_freq = np.fft.rfftfreq(len(data)) * fs
    dfreq = data_freq[1] - data_freq[0]

    noise_psd_interp = noise_psd.interpolate(dfreq)
    noise_psd_interp[noise_psd_interp == 0] = 1.0

    snr = 4 * np.abs(data_fd) ** 2 / noise_psd_interp.value * dfreq
    snr = np.sum(snr[fmin <= data_freq])
    snr = np.sqrt(snr)

    return snr


def _set_missing_params(default, supplied):
    common = set(default).intersection(supplied)
    res = {k: supplied[k] for k in common}
    for k in default.keys() - common:
        res.update({k: default[k]})
    return res


def get_waveform_generator(**waveform_generator_params):
    """Get a waveform generator using
    :meth:`bilby.gw.waveform_generator.WaveformGenerator`

    Args:
        waveform_generator_params: dict
        Keyword arguments to waveform generator
    """
    default_waveform_sampling_params = dict(
        duration=8,
        sampling_frequency=16384,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    )
    default_waveform_approximant_params = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50,
        minimum_frequency=20,
    )

    sampling_params = _set_missing_params(
        default_waveform_sampling_params, waveform_generator_params
    )
    waveform_approximant_params = _set_missing_params(
        default_waveform_approximant_params, waveform_generator_params
    )

    sampling_params["waveform_arguments"] = waveform_approximant_params

    logging.debug("Waveform parameters: {}".format(sampling_params))
    return bilby.gw.waveform_generator.WaveformGenerator(**sampling_params)


def apply_high_pass_filter(signals, sample_params, waveform_generator):
    sos = sig.butter(
        N=8,
        Wn=waveform_generator.waveform_arguments["minimum_frequency"],
        btype="highpass",
        output="sos",
        fs=waveform_generator.sampling_frequency,
    )
    polarization_names = None
    for i, p in enumerate(sample_params):
        polarizations = waveform_generator.time_domain_strain(p)
        if polarization_names is None:
            polarization_names = sorted(polarizations.keys())

        polarizations = np.stack(
            [polarizations[p] for p in polarization_names]
        )
        filtered = sig.sosfiltfilt(sos, polarizations, axis=1)
        signals[i] = filtered
    return signals
