from collections import defaultdict
from pathlib import Path
from typing import List

import bilby
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

from bbhnet.injection.utils import (
    apply_high_pass_filter,
    calc_snr,
    get_waveform_generator,
)
from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.parallelize import AsyncExecutor, as_completed


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


# TODO: maybe put this in the parallelize library
# since its used in multiple places
def load_segment(segment: Segment, *datasets):
    """
    Quick utility function which just wraps a Segment's
    `load` method so that we can execute it in a process
    pool since methods aren't picklable.
    """
    segment.load(*datasets)
    return segment


def inject_into_segment(
    segment: Segment,
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator,
    buffer_: float,
    spacing: float,
    jitter: float,
    priors: bilby.gw.prior.BBHPriorDict,
    sample_rate: float,
    fftlength: float = 2,
    *ifos,
):
    """Helper function to inject_into_timeslide
    to assist in parallelization
    """

    if jitter > spacing:
        raise ValueError(
            "Cannot have jitter greater than spacing: "
            "allows possibility of overlapping signals"
        )

    start = segment.t0 + buffer_
    stop = segment.tf - buffer_

    # determine signal times
    # based on length of segment and spacing;
    # The signal time represents the first sample
    # in the signals generated by project_raw_gw.
    # not to be confused with the t0, which should
    # be the middle sample

    signal_times = np.arange(start, stop, spacing)

    # add random jitter to signal times
    jitter = np.random.uniform(-jitter, jitter, size=len(signal_times))
    signal_times += jitter

    n_samples = len(signal_times)

    # sample prior for this segment
    segment_parameters = priors.sample(n_samples)

    waveform_duration = waveform_generator.duration
    # the center of the sample
    # is geocent time
    segment_parameters["geocent_time"] = signal_times + (waveform_duration / 2)

    # generate raw waveforms
    raw_signals = generate_gw(
        segment_parameters, waveform_generator=waveform_generator
    )

    # dictionary to store
    # gwpy timeseries of background
    raw_ts = {}

    for i, ifo in enumerate(ifos):
        # since we already pre loaded
        # segment the data should be
        # in the cache
        data, times = segment.load(ifo)

        raw_ts[ifo] = TimeSeries(data, times=times)

        # calculate psd for this segment
        psd = raw_ts[ifo].psd(fftlength)

        # project raw waveforms
        signals, snr = project_raw_gw(
            raw_signals,
            segment_parameters,
            waveform_generator,
            ifo,
            get_snr=True,
            noise_psd=psd,
        )

        # store snr
        segment_parameters[f"{ifo}_snr"] = snr

        # loop over signals, injecting them into the
        # raw strain

        for signal_start, signal in zip(signal_times, signals):
            signal_stop = signal_start + len(signal) * (1 / sample_rate)
            signal_times = np.arange(
                signal_start, signal_stop, 1 / sample_rate
            )

            # create gwpy timeseries for signal
            signal = TimeSeries(signal, times=signal_times)

            # inject into raw background
            raw_ts[ifo] = raw_ts[ifo].inject(signal)

    return raw_ts, segment_parameters


def inject_signals_into_timeslide(
    process_ex: AsyncExecutor,
    thread_ex: AsyncExecutor,
    raw_timeslide: TimeSlide,
    out_timeslide: TimeSlide,
    ifos: List[str],
    prior_file: Path,
    spacing: float,
    jitter: float,
    sample_rate: float,
    file_length: int,
    fmin: float,
    waveform_duration: float = 8,
    reference_frequency: float = 20,
    waveform_approximant: float = "IMRPhenomPv2",
    buffer_: float = 0,
    fftlength: float = 2,
):

    """Injects simulated BBH signals into h5 files TimeSlide object that represents
    timeshifted background data. Currently only supports h5 file format.

    Args:
        raw_timeslide: TimeSlide object of raw background data Segments
        out_timeslide: TimeSlide object to store injection Segments
        ifos: list of interferometers corresponding to timeseries
        prior_file: prior file for bilby to sample from
        spacing: seconds between each injection
        jitter: maximum amplitude of random offset to add to signal times
        sample_rate: sampling rate
        file_length: length in seconds of each h5 file
        fmin: Minimum frequency for highpass filter
        waveform_duration: length of injected waveforms
        reference_frequency: reference frequency for generating waveforms
        waveform_approximant: waveform type to inject
        buffer: buffer between beginning and end of segments and waveform
        fftlength: fftlength to use for calculating psd

    Returns:
        Paths to the injected files and the parameter file
    """

    # define a Bilby waveform generator

    # TODO: should sampling rate be automatically inferred
    # from raw data?
    waveform_generator = get_waveform_generator(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=fmin,
        sampling_frequency=sample_rate,
        duration=waveform_duration,
    )

    # initiate prior
    priors = bilby.gw.prior.BBHPriorDict(prior_file)

    # dict to store all parameters
    # of injections
    parameters = defaultdict(list)

    # dictionary to store
    # futures for loading segment data
    load_futures = []

    for segment in raw_timeslide.segments:

        # submit a load job to the process pool
        future = process_ex.submit(load_segment, segment)
        load_futures.append(future)

        # as load jobs are finished
        # submit inject into segment jobs
        inject_futures = []
        for seg in as_completed(load_futures):

            future = process_ex.submit(
                inject_into_segment,
                seg,
                waveform_generator,
                buffer_,
                spacing,
                jitter,
                priors,
                sample_rate,
                fftlength,
                *ifos,
            )

            inject_futures.append(future)

        for inj_datasets, segment_parameters in as_completed(inject_futures):

            # as injection futures complete
            # submit a write job
            times = np.arange(segment.t0, segment.tf, 1 / sample_rate)
            future = thread_ex.submit(
                write_timeseries,
                out_timeslide.path,
                prefix="inj",
                t=times,
                **inj_datasets,
            )

            # append to master parameters dict
            for key, value in segment_parameters.items():
                parameters[key].extend(value)

    # concat parameters for all segments and save
    with h5py.File(out_timeslide.path / "params.h5", "w") as f:
        for k, v in parameters.items():
            f.create_dataset(k, data=v)

    return out_timeslide
