import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import datagen.utils.timeslide_waveforms as utils
import numpy as np
import torch
from datagen.utils.injection import generate_gw
from mldatafind.segments import query_segments
from typeo import scriptify

from aframe.analysis.ledger.injections import (
    InjectionParameterSet,
    LigoResponseSet,
)
from aframe.deploy import condor
from aframe.logging import configure_logging
from aframe.priors.priors import convert_mdc_prior_samples
from ml4gw.gw import (
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
)


@scriptify
def main(
    start: float,
    stop: float,
    shifts: List[float],
    background: Path,
    spacing: float,
    buffer: float,
    waveform_duration: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    output_dir: Path,
    log_file: Optional[Path] = None,
    verbose: bool = False,
):
    """
    Generates the waveforms for a single segment
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file, verbose=verbose)

    prior, detector_frame_prior = prior()

    injection_times = utils.calc_segment_injection_times(
        start,
        stop - max(shifts),  # TODO: should account for uneven last batch too
        spacing,
        buffer,
        waveform_duration,
    )
    n_samples = len(injection_times)
    waveform_size = int(sample_rate * waveform_duration)

    zeros = np.zeros((n_samples,))
    parameters = defaultdict(lambda: zeros.copy())
    parameters["gps_time"] = injection_times
    parameters["shift"] = np.array([shifts for _ in range(n_samples)])

    for ifo in ifos:
        empty = np.zeros((n_samples, waveform_size))
        parameters[ifo.lower()] = empty

    tensors, vertices = get_ifo_geometry(*ifos)
    df = 1 / waveform_duration
    try:
        background = next(background.iterdir())
    except StopIteration:
        raise ValueError(f"No files in background data directory {background}")
    psds = utils.load_psds(background, ifos, sample_rate=sample_rate, df=df)

    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    num_injections, idx = 0, 0
    rejected_params = InjectionParameterSet()
    while n_samples > 0:
        params = prior.sample(n_samples)
        params = convert_mdc_prior_samples(params)
        waveforms = generate_gw(
            params,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            detector_frame_prior,
        )
        polarizations = {
            "cross": torch.Tensor(waveforms[:, 0, :]),
            "plus": torch.Tensor(waveforms[:, 1, :]),
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
        # TODO: compute individual ifo snr so we can store that data
        snrs = compute_network_snr(projected, psds, sample_rate, highpass)
        snrs = snrs.numpy()

        # add all snrs: masking will take place in for loop below
        params["snr"] = snrs
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
        start, stop = idx, idx + num_accepted
        for key, value in params.items():
            # TODO: at this point, start using the
            # __dataclass_fields__ instead. Maybe
            # arm the metaclasses with a .parameters
            # method to make the logic of getting those simpler
            if key not in (
                "mass_ratio",
                "chirp_mass",
                "luminosity_distance",
                "chirp_distance",
            ):
                parameters[key][start:stop] = value[mask]

        # do the same for our accepted projected waveforms
        projected = projected[mask].numpy()
        for i, ifo in enumerate(ifos):
            key = ifo.lower()
            parameters[key][start:stop] = projected[:, i]

        # subtract off the number of samples we accepted
        # from the number we'll need to sample next time,
        # that way we never overshoot our number of desired
        # accepted samples and therefore risk overestimating
        # our total number of injections
        idx += num_accepted
        n_samples -= num_accepted

    parameters["sample_rate"] = sample_rate
    parameters["duration"] = waveform_duration
    parameters["num_injections"] = num_injections

    response_set = LigoResponseSet(**parameters)
    waveform_fname = output_dir / "waveforms.h5"
    utils.io_with_blocking(response_set.write, waveform_fname)

    # For MDC dataset we don't need to save rejected parameters
    # since we use a hopeless snr threshold of 0

    rejected_fname = output_dir / "rejected-parameters.h5"
    utils.io_with_blocking(rejected_params.write, rejected_fname)

    # TODO: compute probability of all parameters against
    # source and all target priors here then save them somehow
    return waveform_fname, rejected_fname


# until typeo update gets in just take all the same parameter as main
@scriptify
def deploy(
    start: float,
    stop: float,
    state_flag: str,
    Tb: float,
    shifts: Iterable[float],
    spacing: float,
    buffer: float,
    min_segment_length: float,
    waveform_duration: float,
    prior: str,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    psd_length: float,
    outdir: Path,
    datadir: Path,
    logdir: Path,
    accounting_group_user: str,
    accounting_group: str,
    request_memory: int = 6000,
    request_disk: int = 1024,
    verbose: bool = False,
    force_generation: bool = False,
) -> None:
    # define some directories:
    # outdir: where we'll write the temporary
    #    files created in each condor job
    # writedir: where we'll write the aggregated
    #    aggregated outputs from each of the
    #    temporary files in outdir
    # condordir: where we'll write the submit file,
    #    queue parameters file, and the log, out, and
    #    err files from each submitted job
    outdir = outdir / "timeslide_waveforms"
    writedir = datadir / "test"
    condordir = outdir / "condor"
    for d in [outdir, writedir, condordir, logdir]:
        d.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "timeslide_waveforms.log", verbose=verbose)

    # check to see if any of the target files are
    # missing or if we've indicated to force
    # generation even if they are
    for fname in ["waveforms.h5", "rejected-parameters.h5"]:
        if not (writedir / fname).exists() or force_generation:
            break
    else:
        # if everything exists and we're not forcing
        # generation, short-circuit here
        logging.info(
            "Timeslide waveform and rejected parameters files "
            "already exist in {} and force_generation is off, "
            "exiting".format(writedir)
        )
        return

    # query segments and calculate shifts required
    # to accumulate desired background livetime
    state_flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    segments = query_segments(state_flags, start, stop, min_segment_length)
    shifts_required = utils.get_num_shifts(segments, Tb, max(shifts))

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,shift\n"
    for start, stop in segments:
        for i in range(shifts_required):
            # TODO: make this more general
            shift = [(i + 1) * shift for shift in shifts]
            shift = " ".join(map(str, shift))
            # add psd_length to account for the burn in of psd calculation
            parameters += f"{start + psd_length},{stop},{shift}\n"

    # TODO: have typeo do this CLI argument construction?
    arguments = "--start $(start) --stop $(stop) --shifts $(shift) "
    arguments += f"--background {datadir / 'train' / 'background'} "
    arguments += f"--spacing {spacing} --buffer {buffer} "
    arguments += f"--waveform-duration {waveform_duration} "
    arguments += f"--minimum-frequency {minimum_frequency} "
    arguments += f"--reference-frequency {reference_frequency} "
    arguments += f"--sample-rate {sample_rate} "
    arguments += f"--waveform-approximant {waveform_approximant} "
    arguments += f"--highpass {highpass} --snr-threshold {snr_threshold} "
    arguments += f"--ifos {' '.join(ifos)} "
    arguments += f"--prior {prior} "
    arguments += f"--output-dir {outdir}/tmp-$(ProcID) "
    arguments += f"--log-file {logdir}/$(ProcID).log "

    # create submit file by hand: pycondor doesn't support
    # "queue ... from" syntax
    subfile = condor.make_submit_file(
        executable="generate-timeslide-waveforms",
        name="timeslide_waveforms",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory=request_memory,
        request_disk=request_disk,
    )
    dag_id = condor.submit(subfile)
    logging.info(f"Launching waveform generation jobs with dag id {dag_id}")
    condor.watch(dag_id, condordir)

    # once all jobs are done, merge the output files
    waveform_fname = writedir / "waveforms.h5"
    waveform_files = list(outdir.rglob("waveforms.h5"))
    logging.info(f"Merging output waveforms to file {waveform_fname}")
    LigoResponseSet.aggregate(waveform_files, waveform_fname, clean=True)

    params_fname = writedir / "rejected-parameters.h5"
    param_files = list(outdir.rglob("rejected-parameters.h5"))
    logging.info(f"Merging rejected parameters to file {params_fname}")
    InjectionParameterSet.aggregate(param_files, params_fname, clean=True)

    for dirname in outdir.glob("tmp-*"):
        shutil.rmtree(dirname)

    logging.info("Timeslide waveform generation complete")
