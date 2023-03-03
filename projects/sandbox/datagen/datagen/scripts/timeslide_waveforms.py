import logging
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List

import datagen.utils.timeslide_waveforms as utils
import h5py
import numpy as np
import torch
from datagen.utils.injection import generate_gw
from mldatafind.segments import query_segments
from typeo import scriptify

from bbhnet.logging import configure_logging
from ml4gw.gw import (
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
)

# re for extracting cluster id from condor_submit output
# stolen from pyomicron:
# https://github.com/ML4GW/pyomicron/blob/master/omicron/condor.py
re_dagman_cluster = re.compile(r"(?<=submitted\sto\scluster )[0-9]+")


@scriptify
def main(
    start: float,
    stop: float,
    hanford_background: Path,
    livingston_background: Path,
    spacing: float,
    buffer: float,
    waveform_duration: float,
    cosmology: Callable,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    output_fname: Path,
):
    """
    Generates the waveforms for a single segment
    """

    cosmology = cosmology()
    prior, detector_frame_prior = prior(cosmology)

    injection_times = utils.calc_segment_injection_times(
        start, stop, spacing, buffer, waveform_duration
    )
    n_samples = len(injection_times)

    signals = torch.Tensor()
    parameters = defaultdict(list)
    parameters["geocent_time"] = injection_times

    tensors, vertices = get_ifo_geometry(*ifos)
    df = 1 / waveform_duration
    psds = utils.load_psds(
        hanford_background,
        livingston_background,
        sample_rate=sample_rate,
        df=df,
    )
    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    n_rejected = 0
    while len(signals) < n_samples:

        params = prior.sample(n_samples)
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
        mask = snrs > snr_threshold

        projected = projected[mask]
        n_rejected += np.sum(~mask)

        signals = torch.cat((signals, projected))
        for key, value in params.items():
            parameters[key].extend(list(value[mask]))

    signals = signals[:n_samples]
    for key, value in parameters.items():
        parameters[key] = value[:n_samples]

    with h5py.File(output_fname, "w") as f:
        f.create_dataset("signals", data=signals)
        for k, v in parameters.items():
            f.create_dataset(k, data=v)

        f.attrs.update(
            {
                "n_rejected": n_rejected,
            }
        )
    return output_fname


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
    cosmology: str,
    waveform_duration: float,
    prior: str,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    datadir: Path,
    logdir: Path,
    accounting_group_user: str,
    accounting_group: str,
    request_memory: int = 6000,
    request_disk: int = 1024,
    verbose: bool = False,
):

    outdir = datadir / "timeslide_waveforms"

    outdir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "timeslide_waveforms.log", verbose=verbose)
    hanford_background = datadir / "H1_background.h5"
    livingston_background = datadir / "L1_background.h5"

    # where condor info and sub files will live
    condor_dir = outdir / "condor"
    condor_dir.mkdir(exist_ok=True, parents=True)

    # query segments and calculate shifts required
    # to accumulate desired background livetime

    state_flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    segments = query_segments(state_flags, start, stop, min_segment_length)
    shifts_required = utils.calc_shifts_required(segments, Tb, max(shifts))

    # TODO: does this logic generalize to negative shifts?
    max_shift = max(shifts) * shifts_required

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    with open(condor_dir / "segments.txt", "w") as f:
        for start, stop in segments:
            for i in range(shifts_required):
                f.write(f"{start},{stop - max_shift}\n")

    executable = shutil.which("generate-timeslide-waveforms")

    # TODO: have typeo do this CLI argument construction?
    arguments = "--start $(start) --stop $(stop) "
    arguments += f"--hanford-background {hanford_background} "
    arguments += f"--livingston-background {livingston_background} "
    arguments += f"--spacing {spacing} --buffer {buffer} "
    arguments += f"--waveform-duration {waveform_duration} "
    arguments += f"--minimum-frequency {minimum_frequency} "
    arguments += f"--reference-frequency {reference_frequency} "
    arguments += f"--sample-rate {sample_rate} "
    arguments += f"--waveform-approximant {waveform_approximant} "
    arguments += f"--highpass {highpass} --snr-threshold {snr_threshold} "
    arguments += f"--ifos {' '.join(ifos)} "
    arguments += f"--prior {prior} --cosmology {cosmology} "
    arguments += f"--output-fname {outdir}/$(ProcID).hdf5 "

    # create submit file by hand: pycondor doesn't support
    # "queue ... from" syntax
    subfile = utils.create_submit_file(
        executable,
        condor_dir,
        accounting_group,
        accounting_group_user,
        request_memory,
        request_disk,
        arguments,
    )

    subfile_path = condor_dir / "timeslide_waveforms.submit"
    with open(subfile_path, "w") as f:
        f.write(subfile)

    # launch the jobs via condor_submit,
    # extract the dag id from the output,
    # and monitor the dag with condor_watch_q
    condor_submit = shutil.which("condor_submit")
    out = subprocess.check_output(
        [str(condor_submit), str(condor_dir / "timeslide_waveforms.submit")]
    ).decode("utf-8")

    dagid = int(re_dagman_cluster.search(out).group())
    cwq = shutil.which("condor_watch_q")

    logging.info("Launching waveform generation jobs")
    subprocess.check_call(
        [
            cwq,
            "-exit",
            "all,done,0",
            "-exit",
            "any,held,1",
            "-clusters",
            str(dagid),
        ]
    )

    logging.info("Merging output files")
    # once all jobs are done, merge the output files
    utils.merge_output(outdir)

    logging.info("Timeslide waveform generation complete")
