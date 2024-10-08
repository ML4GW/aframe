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
    coalescence_time: float,
    highpass: float,
    snr_threshold: float,
    psd: Union[Path, torch.Tensor],
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
    num_injections, idx = 0, 0
    rejected_params = InjectionParameterSet()
    while num_signals > 0:
        params = prior.sample(num_signals)
        if not detector_frame_prior:
            params = convert_to_detector_frame(params)
        if num_signals == 1:
            params = {k: params[k] for k in prior.keys() if k in params}

        polarization_set = WaveformPolarizationSet.from_parameters(
            BilbyParameterSet(**params),
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            coalescence_time,
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
        ifo_snrs = compute_ifo_snr(projected, psd, sample_rate, highpass)
        snrs = ifo_snrs**2
        snrs = snrs.sum(axis=-1) ** 0.5
        snrs = snrs.numpy()

        # add all snrs: masking will take place in for loop below
        params["snr"] = snrs
        params["ifo_snrs"] = ifo_snrs.numpy()

        num_injections += len(snrs)
        mask = snrs >= snr_threshold

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
        start, end = idx, idx + num_accepted
        for key, value in params.items():
            parameters[key][start:end] = value[mask]

        # insert either the projected waveforms or the raw waveforms

        signals = projected[mask].numpy()

        for i, ifo in enumerate(ifos):
            key = ifo.lower()
            parameters[key][start:end] = signals[:, i]

        # subtract off the number of samples we accepted
        # from the number we'll need to sample next time,
        # that way we never overshoot our number of desired
        # accepted samples and therefore risk overestimating
        # our total number of injections
        idx += num_accepted
        num_signals -= num_accepted

    parameters["coalescence_time"] = coalescence_time
    parameters["sample_rate"] = sample_rate
    parameters["duration"] = waveform_duration
    parameters["num_injections"] = num_injections
    parameters["ifos"] = ifos
    return parameters, rejected_params
