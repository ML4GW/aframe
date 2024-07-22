from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator
from data.waveforms.utils import convert_to_detector_frame
from ledger.injections import InjectionParameterSet, _WaveformGenerator

from ml4gw.gw import compute_ifo_snr, compute_observed_strain, get_ifo_geometry

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
    psds: torch.Tensor,
) -> Tuple[ResponseSetFields, InjectionParameterSet]:
    # get the detector tensors and vertices
    # for projecting our waveforms
    tensors, vertices = get_ifo_geometry(*ifos)

    # instantiate a waveform generator whose
    # call method will generate raw polarizations
    _generator = WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": waveform_approximant,
            "reference_frequency": reference_frequency,
            "minimum_frequency": minimum_frequency,
        },
    )

    generator = _WaveformGenerator(
        _generator, sample_rate, waveform_duration, coalescence_time
    )

    # create a dictionary to store accepted
    # parameters, waveforms, and metadata
    zeros = np.zeros((num_signals,))
    parameters = defaultdict(lambda: zeros.copy())

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

        # TODO: can encapsulate this in a
        # WaveformSet.from_parameters method
        params_list = [dict(zip(params, col)) for col in zip(*params.values())]
        polarizations = {
            "cross": torch.zeros((len(params_list), waveform_size)),
            "plus": torch.zeros((len(params_list), waveform_size)),
        }

        for i, polars in enumerate(map(generator, params_list)):
            for key, value in polars.items():
                polarizations[key][i] = torch.Tensor(value)

        projected = compute_observed_strain(
            torch.Tensor(params["dec"]),
            torch.Tensor(params["psi"]),
            torch.Tensor(params["ra"]),
            tensors,
            vertices,
            sample_rate,
            **polarizations,
        )
        ifo_snrs = compute_ifo_snr(projected, psds, sample_rate, highpass)
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
        for key in InjectionParameterSet.__dataclass_fields__:
            rejected[key] = params[key][~mask]
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
