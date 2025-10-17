from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from ml4gw.gw import compute_ifo_snr, compute_observed_strain, get_ifo_geometry

from data.waveforms.utils import convert_to_detector_frame, load_psds
from ledger.injections import (
    BilbyParameterSet,
    InjectionParameterSet,
    WaveformPolarizationSet,
)

ResponseSetFields = Dict[str, Union[np.ndarray, float]]


def rejection_sample(
    num_signals: int,
    prior: Callable,
    ifos: List[str],
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    right_pad: float,
    highpass: float,
    lowpass: float,
    snr_threshold: float,
    psd: Union[Path, torch.Tensor],
    max_num_samples: int,
) -> Tuple[ResponseSetFields, InjectionParameterSet]:
    # get the detector tensors and vertices
    # for projecting our waveforms
    tensors, vertices = get_ifo_geometry(*ifos)

    # create a dictionary to store accepted
    # parameters, waveforms, and metadata
    zeros = np.zeros((num_signals,))
    parameters = defaultdict(lambda: zeros.copy())

    # ifo snr array will be 2 dimensional
    parameters["ifo_snrs"] = np.zeros((num_signals, len(ifos)))

    # allocate memory for our waveforms up front
    waveform_size = int(sample_rate * waveform_duration)
    for ifo in ifos:
        empty = np.zeros((num_signals, waveform_size))
        parameters[ifo.lower()] = empty

    prior, detector_frame_prior = prior()

    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    num_injections, total_accepted = 0, 0
    rejected_params = InjectionParameterSet()
    # Start by simulating the desired number of accepted signals
    num_samples = num_signals
    while total_accepted < num_signals:
        params = prior.sample(num_samples)
        if not detector_frame_prior:
            params = convert_to_detector_frame(params)
        if num_samples == 1:
            params = {k: params[k] for k in prior.keys() if k in params}

        polarization_set = WaveformPolarizationSet.from_parameters(
            BilbyParameterSet(**params),
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            right_pad,
        )
        polarizations = {
            "cross": torch.Tensor(polarization_set.cross),
            "plus": torch.Tensor(polarization_set.plus),
        }

        projected = compute_observed_strain(
            torch.Tensor(params["dec"]),
            torch.Tensor(params["psi"]),
            torch.Tensor(params["ra"]),
            tensors,
            vertices,
            sample_rate,
            **polarizations,
        )

        # Load/calculate psds if not given explicitly
        if isinstance(psd, Path):
            df = 1 / waveform_duration
            psd = load_psds(psd, ifos, df)

        # compute both individual ifo snrs and network snr
        ifo_snrs = compute_ifo_snr(
            projected, psd, sample_rate, highpass, lowpass
        )
        snrs = ifo_snrs**2
        snrs = snrs.sum(axis=-1) ** 0.5
        snrs = snrs.numpy()

        # add all snrs: masking will take place in for loop below
        params["snr"] = snrs
        params["ifo_snrs"] = ifo_snrs.numpy()

        num_injections += len(snrs)
        mask = snrs >= snr_threshold

        num_accepted = mask.sum()
        total_accepted += num_accepted
        # If we've generated more signals than we need,
        # figure out where to cut off the arrays by
        # identifying the index at which we reach
        # the target number of accepted signals.
        if num_signals < total_accepted:
            target_accepted = num_signals - (total_accepted - num_accepted)
            idx = np.where(np.cumsum(mask) == target_accepted)[0][0] + 1
            mask = mask[:idx]
            projected = projected[:idx]
            params = {k: v[:idx] for k, v in params.items()}
            num_accepted = mask.sum()
            total_accepted = num_signals

        # first record any parameters that were
        # rejected during sampling to a separate object
        rejected = {}
        for key, attr in InjectionParameterSet.__dataclass_fields__.items():
            if attr.metadata["kind"] == "parameter":
                rejected[key] = params[key][~mask]

        # add the ifo metadata attribute
        rejected["ifos"] = ifos
        rejected = InjectionParameterSet(**rejected)
        rejected_params.append(rejected)

        # if nothing got accepted, try again
        num_accepted = mask.sum()
        if num_accepted == 0:
            continue

        # insert our accepted parameters into the output array
        start, end = total_accepted - num_accepted, total_accepted
        for key, value in params.items():
            parameters[key][start:end] = value[mask]

        # insert either the projected waveforms or the raw waveforms

        signals = projected[mask].numpy()

        for i, ifo in enumerate(ifos):
            key = ifo.lower()
            parameters[key][start:end] = signals[:, i]

        # Estimate how many more samples need to be generated
        # to reach our desired number of accepted samples.
        samples_remaining = num_signals - total_accepted
        acceptance_rate = num_accepted / num_samples
        # We `continue` above if `num_accepted == 0`, so
        # no need to worry above division by zero here.
        num_samples = int(np.ceil(samples_remaining / acceptance_rate))
        # To make sure we don't exceed memory limits,
        # don't generate more than `max_num_samples`
        num_samples = min(num_samples, max_num_samples)

        # To make sure we don't waste time repeatedly
        # generating a small number of samples, always
        # generate at least the initial number of signals.
        # TODO: Is there a more efficient lower bound?
        num_samples = max(num_samples, num_signals)

    parameters["right_pad"] = right_pad
    parameters["sample_rate"] = sample_rate
    parameters["duration"] = waveform_duration
    parameters["num_injections"] = num_injections
    parameters["ifos"] = ifos
    return parameters, rejected_params
