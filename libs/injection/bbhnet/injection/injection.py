import logging
from collections.abc import Iterable
from pathlib import Path

import bilby
import h5py
import numpy as np
import scipy.signal as sig
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from gwpy.timeseries import TimeSeries


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


def generate_gw(
    sample_params, waveform_generator=None, **waveform_generator_params
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.

    Args:
        sample_params: dictionary of GW parameters
        waveform_generator: bilby.gw.WaveformGenerator with appropriate params
        waveform_generator_params: keyword arguments to
        :meth:`bilby.gw.WaveformGenerator`

    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples. The first polarization is
        always plus and the second is always cross
    """

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_samples = len(sample_params)

    waveform_generator = waveform_generator or get_waveform_generator(
        **waveform_generator_params
    )

    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration
    waveform_size = int(sample_rate * waveform_duration)

    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    filtered_signal = apply_high_pass_filter(
        signals, sample_params, waveform_generator
    )
    return filtered_signal


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


def project_raw_gw(
    raw_waveforms,
    sample_params,
    waveform_generator,
    ifo,
    get_snr=False,
    noise_psd=None,
):
    """Project a raw gravitational wave onto an intterferometer

    Args:
        raw_waveforms: the plus and cross polarizations of a list of GWs
        sample_params: dictionary of GW parameters
        waveform_generator: the waveform generator that made the raw GWs
        ifo: interferometer
        get_snr: return the SNR of each sample
        noise_psd: background noise PSD used to calculate SNR the sample

    Returns:
        An (n_samples, waveform_size) array containing the GW signals as they
        would appear in the given interferometer with the given set of sample
        parameters. If get_snr=True, also returns a list of the SNR associated
        with each signal
    """

    polarizations = {
        "plus": raw_waveforms[:, 0, :],
        "cross": raw_waveforms[:, 1, :],
    }

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_sample = len(sample_params)

    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration
    waveform_size = int(sample_rate * waveform_duration)

    signals = np.zeros((n_sample, waveform_size))
    snr = np.zeros(n_sample)

    ifo = bilby.gw.detector.get_empty_interferometer(ifo)
    for i, p in enumerate(sample_params):

        # For less ugly function calls later on
        ra = p["ra"]
        dec = p["dec"]
        geocent_time = p["geocent_time"]
        psi = p["psi"]

        # Generate signal in IFO
        signal = np.zeros(waveform_size)
        for mode, polarization in polarizations.items():
            # Get ifo response
            response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
            signal += response * polarization[i]

        # Total shift = shift to trigger time + geometric shift
        dt = waveform_duration / 2.0
        dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)
        signal = np.roll(signal, int(np.round(dt * sample_rate)))

        # Calculate SNR
        if noise_psd is not None:
            if get_snr:
                snr[i] = calc_snr(signal, noise_psd, sample_rate)

        signals[i] = signal
    if get_snr:
        return signals, snr
    return signals


def inject_signals(
    frame_files: Iterable[str],
    channels: [str],
    ifos: [str],
    prior_file: str,
    n_samples: int,
    outdir: str,
    fmin: float = 20,
    waveform_duration: float = 8,
    snr_range: Iterable[float] = [25, 50],
):

    """Injects simulated BBH signals into a frame, or set of corresponding
    frames from different interferometers. Frames should have the same
    start/stop time and the same sample rate

    Args:
        frame_files: list of paths to frames to be injected
        channels: channel names of the strain data in each frame
        ifos: list of interferometers corresponding to frames, e.g., H1, L1
        prior_file: prior file for bilby to sample from
        n_samples: number of signal to inject
        outdir: output directory to which injected frames will be written
        fmin: Minimum frequency for highpass filter
        waveform_duration: length of injected waveforms
        snr_range: desired signal SNR range

    Returns:
        Paths to the injected frames and the parameter file
    """

    strains = [
        TimeSeries.read(frame, ch) for frame, ch in zip(frame_files, channels)
    ]

    logging.info("Read strain from frame files")

    span = set([strain.span for strain in strains])
    if len(span) != 1:
        raise ValueError(
            "Frame files {} and {} have different durations".format(
                *frame_files
            )
        )

    frame_start, frame_stop = next(iter(span))
    frame_duration = frame_stop - frame_start

    sample_rate = set([int(strain.sample_rate.value) for strain in strains])
    if len(sample_rate) != 1:
        raise ValueError(
            "Frame files {} and {} have different sample rates".format(
                *frame_files
            )
        )

    sample_rate = next(iter(sample_rate))
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
    sample_params["geocent_time"] = signal_times

    signals_list = []
    snr_list = []
    for strain, channel, ifo in zip(strains, channels, ifos):

        # calculate the PSD
        strain_psd = strain.psd(fftlength)

        # generate GW waveforms
        raw_signals = generate_gw(
            sample_params,
            waveform_generator=waveform_generator,
        )

        signals, snr = project_raw_gw(
            raw_signals,
            sample_params,
            waveform_generator,
            ifo,
            get_snr=True,
            noise_psd=strain_psd,
        )

        signals_list.append(signals)
        snr_list.append(snr)

    old_snr = np.sqrt(np.sum(np.square(snr_list), axis=0))
    new_snr = np.random.uniform(snr_range[0], snr_range[1], len(snr_list[0]))

    signals_list = [
        signals * (new_snr / old_snr)[:, None] for signals in signals_list
    ]
    sample_params["luminosity_distance"] = (
        sample_params["luminosity_distance"] * old_snr / new_snr
    )
    snr_list = [snr * new_snr / old_snr for snr in snr_list]

    outdir = Path(outdir)
    frame_out_paths = [outdir / f.name for f in map(Path, frame_files)]

    for strain, signals, frame_path in zip(
        strains, signals_list, frame_out_paths
    ):
        for i in range(n_samples):
            idx1 = int(
                (signal_times[i] - waveform_duration / 2.0) * sample_rate
            )
            idx2 = idx1 + waveform_duration * sample_rate
            strain[idx1:idx2] += signals[i]

        strain.write(frame_path)

    # Write params and similar to output file
    param_file = outdir / f"param_file_{frame_start}-{frame_stop}.h5"
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

    return frame_out_paths, param_file
