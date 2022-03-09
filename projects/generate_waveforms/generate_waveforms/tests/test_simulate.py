#!/usr/bin/env python
# coding: utf-8
import logging
import os

import bilby
import h5py
import numpy as np
import scipy.signal as sig
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole


def generate_gw(
    sample_params,
    ifo,
    waveform_generator=None,
):
    """Generate gravitational-wave events

    Arguments:
    - sample_params: dictionary of GW parameters
    - ifo: interferometer
    - waveform_generator: bilby.gw.WaveformGenerator with appropriate params
    """

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_sample = len(sample_params)

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

    ifo = bilby.gw.detector.get_empty_interferometer(ifo)
    b, a = sig.butter(
        N=8,
        Wn=waveform_generator.waveform_arguments["minimum_frequency"],
        btype="highpass",
        fs=waveform_generator.sampling_frequency,
    )
    for i, p in enumerate(sample_params):

        # For less ugly function calls later on
        ra = p["ra"]
        dec = p["dec"]
        geocent_time = p["geocent_time"]
        psi = p["psi"]

        # Generate signal in IFO
        polarizations = waveform_generator.time_domain_strain(p)
        signal = np.zeros(waveform_size)
        for mode, polarization in polarizations.items():
            # Get ifo response
            response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
            signal += response * sig.filtfilt(b, a, polarization)

        # Total shift = shift to trigger time + geometric shift
        dt = waveform_duration / 2.0
        dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)
        signal = np.roll(signal, int(np.round(dt * sample_rate)))

        signals[i] = signal

    return signals


def main(
    prior_file: str,
    n_samples: int,
    outdir: str,
    waveform_duration: float = 8,
    sample_rate: float = 4096,
):

    """Simulates a set of BBH waveforms that can be added to background

    Arguments:
    - prior_file: prior file for bilby to sample from
    - n_samples: number of signal to inject
    - outdir: output directory to which signals will be written
    - waveform_duration: length of injected waveforms
    - sample_rate: sample rate of the signal in Hz
    """

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(n_samples))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
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
    # Random times from O3
    signal_times = np.random.randint(1238166018, 1269363618, n_samples)
    sample_params["geocent_time"] = signal_times

    H1_signals = generate_gw(
        sample_params,
        "H1",
        waveform_generator=waveform_generator,
    )

    L1_signals = generate_gw(
        sample_params,
        "L1",
        waveform_generator=waveform_generator,
    )

    signals = np.stack([H1_signals, L1_signals], axis=1)

    # Write params and similar to output file
    prior_name = os.path.basename(prior_file)[:-6]
    signal_file = os.path.join(
        outdir, f"signal_file_{prior_name}-{waveform_duration}.h5"
    )

    with h5py.File(signal_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        for k, v in sample_params.items():
            f.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
            }
        )


if __name__ == "__main__":
    main()
