import itertools
import logging
import time
from concurrent.futures import FIRST_EXCEPTION, TimeoutError, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import bilby
import gwdatafind
import h5py
import numpy as np
import torch
from gwpy.segments import (
    DataQualityDict,
    Segment,
    SegmentList,
    SegmentListDict,
)
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from typeo import scriptify

from bbhnet.injection import generate_gw, inject_waveforms
from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor
from ml4gw.gw import compute_ifo_snr, compute_observed_strain, get_ifo_geometry


def download_data(
    ifos: Iterable[str],
    frame_type: str,
    channel: str,
    sample_rate: float,
    start: float,
    stop: float,
) -> TimeSeriesDict:
    data = TimeSeriesDict()
    for ifo in ifos:
        files = gwdatafind.find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=start,
            gpsend=stop,
            urltype="file",
        )
        data[ifo] = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=start, end=stop, nproc=4
        )
    return data.resample(sample_rate)


def get_params(
    priors: bilby.gw.prior.PriorDict,
    n_slides: int,
    signal_times: np.ndarray,
    jitter: float,
    waveform_duration: float,
) -> List[Dict[str, List[float]]]:
    n_signals = len(signal_times)
    params = priors.sample(n_signals * n_slides)

    shift_parameters = []
    for i in range(n_slides):
        slc = slice(i * n_signals, (i + 1) * n_signals)
        param = {p: v[slc] for p, v in params.items()}

        jit = np.random.uniform(-jitter, jitter, size=n_signals)
        param["geocent_time"] = signal_times + jit + waveform_duration / 2

        shift_parameters.append(param)
    return shift_parameters


def generate_waveforms(
    parameters,
    minimum_frequency,
    reference_frequency,
    sample_rate,
    waveform_duration,
    waveform_approximant,
):
    signals = generate_gw(
        parameters,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
    )
    return signals, parameters


@scriptify
def main(
    start: int,
    stop: int,
    logdir: Path,
    datadir: Path,
    prior_file: Path,
    spacing: float,
    jitter: float,
    buffer_: float,
    n_slides: int,
    shifts: Iterable[float],
    ifos: Iterable[str],
    file_length: int,
    minimum_frequency: float,
    highpass: float,
    sample_rate: float,
    frame_type: str,
    channel: str,
    min_segment_length: Optional[float] = None,
    waveform_duration: float = 8,
    reference_frequency: float = 20,
    waveform_approximant: str = "IMRPhenomPv2",
    fftlength: float = 2,
    state_flag: Optional[str] = None,
    verbose: bool = False,
):
    """Generates timeslides of background and background + injections.
    Timeslides are generated on a per segment basis: First, science segments
    are queried for each ifo and coincidence is performed.
    To create a timeslide, each continuous segment is circularly shifted.

    Args:
        start: starting GPS time of time period to analyze
        stop: ending GPS time of time period to analyze
        outdir: base directory where all timeslide directories will be created
        prior_file: a .prior file containing the priors for the GW simulation
        spacing: spacing between consecutive injections
        n_slides: number of timeslides
        shift:
            List of shift multiples for each ifo. Will create n_slides
            worth of shifts, at multiples of shift. If 0 is passed,
            will not shift this ifo for any slide.
        ifos: List interferometers
        file_length: length in seconds of each separate file
        minimum_frequency: minimum_frequency used for waveform generation
        highpass: frequency at which data is highpassed
        sample_rate: sample rate
        frame_type: frame type for data discovery
        channel: strain channel to analyze
        waveform_duration: length of injected waveforms
        reference_frequency: reference frequency for generating waveforms
        waveform_approximant: waveform model to inject
        fftlength: fftlength for calculating psd
        state_flag: name of segments to query from segment database
    """

    logdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    configure_logging(logdir / "timeslide_injections.log", verbose)

    # if state_flag is passed, query segments for each ifo.
    # A certificate is needed for this, see X509 instructions on
    # https://computing.docs.ligo.org/guide/auth/#ligo-x509
    logging.info("Querying segments")
    if state_flag:
        # query science segments
        segments = DataQualityDict.query_dqsegdb(
            [f"{ifo}:{state_flag}" for ifo in ifos],
            start,
            stop,
        )

        # convert DQ dict to SegmentList Dict
        segments = SegmentListDict({k: v.active for k, v in segments.items()})

        # intersect segments
        intersection = segments.intersection(segments.keys())
    else:
        # not considering segments so
        # make intersection from start to stop
        intersection = SegmentList([Segment(start, stop)])

    total_length = sum([j - i for i, j in intersection])
    logging.info(
        "Querying {} segments of data totalling {} worth of data".format(
            len(intersection), total_length
        )
    )

    if not prior_file.is_absolute():
        prior_file = Path(__file__).resolve().parent / prior_file
    priors = bilby.gw.prior.BBHPriorDict(str(prior_file))

    min_segment_length = min_segment_length or 0
    total_slides = n_slides ** (len([i for i in shifts if i]))
    max_shift = int(max(shifts) * n_slides * sample_rate)

    process_pool = AsyncExecutor(4, thread=False)
    thread_pool = AsyncExecutor(2, thread=True)

    # get tensors and vertices for requested ifos
    tensors, vertices = get_ifo_geometry(*ifos)

    with process_pool, thread_pool:
        for segment_start, segment_stop in intersection:
            if segment_stop - segment_start < min_segment_length:
                continue

            # begin the download of data in a separate thread
            logging.debug(
                "Beginning download of segment {}-{}".format(
                    segment_start, segment_stop
                )
            )
            download_future = thread_pool.submit(
                download_data,
                ifos,
                frame_type,
                channel,
                sample_rate,
                segment_start,
                segment_stop,
            )
            download_future.add_done_callback(
                lambda f: logging.debug(
                    "Completed download of segment {}-{}".format(
                        segment_start, segment_stop
                    )
                )
            )

            # generate timestamps of trigger times for each waveform.
            # Jitter will get added in `get_params`.
            # take off min segment length from end of segment stop
            # since we end up cutting some data off to make sure each timeslide
            # has the same length of data
            signal_start = segment_start + buffer_
            signal_stop = segment_stop - buffer_ - max(shifts) * n_slides
            signal_times = np.arange(signal_start, signal_stop, spacing)

            # sample dictionary of params for each timeslide
            logging.debug(
                "Sampling {} waveform parameters".format(
                    total_slides * len(signal_times)
                )
            )
            parameters = get_params(
                priors, total_slides, signal_times, jitter, waveform_duration
            )

            # generate each timeslide's waveforms in parallel
            # wrap with a lambda function to keep the parameters
            logging.debug("Launching waveform generation in parallel")
            waveform_it = process_pool.imap(
                generate_waveforms,
                parameters,
                minimum_frequency=minimum_frequency,
                reference_frequency=reference_frequency,
                sample_rate=sample_rate,
                waveform_duration=waveform_duration,
                waveform_approximant=waveform_approximant,
            )

            # wait until the download has completed to move on
            while True:
                try:
                    background = download_future.result(timeout=1e-3)
                    break
                except TimeoutError:
                    time.sleep(1e-2)

            ranges = [range(n_slides) for i in shifts if i]
            shift_iterator = itertools.product(*ranges)
            futures = []
            for waveforms, parameters in waveform_it:
                # convoluted way to get our shifts, including the 0 values
                # that wouldn't have been included in the shift_iterator
                idx = iter(next(shift_iterator))
                ts_shifts = []
                for shift in shifts:
                    shift = shift if shift == 0 else next(idx) * shift
                    ts_shifts.append(float(shift))

                # create all the directories we'll
                # need for our various timeslides
                shift_str = [f"{i[0]}{j}" for i, j in zip(ifos, ts_shifts)]
                shift_str = "-".join(shift_str)
                logging.debug(
                    "Creating timeslide for segment {}-{} "
                    "with shifts {}".format(
                        segment_start, segment_stop, shift_str
                    )
                )
                root = datadir / f"dt-{shift_str}"
                root.mkdir(exist_ok=True, parents=True)

                raw_ts = TimeSlide.create(root=root, field="background")
                injection_ts = TimeSlide.create(root=root, field="injection")

                # create the appropriate shifts for each interferomter
                background_data = {}
                for shift, ifo in zip(ts_shifts, ifos):
                    shift = int(shift * sample_rate)
                    slc = slice(shift, -max_shift + shift)
                    background_data[ifo] = background[ifo].value[slc]

                    if shift == 0:
                        # TODO: what happens if all are shifted? Is this
                        # a possibility that we want to entertain?
                        times = background[ifo].times.value[slc]

                # submit this data as a write job to the process pool
                future = process_pool.submit(
                    h5.write_timeseries,
                    raw_ts.path,
                    prefix="raw",
                    t=times,
                    **background_data,
                )
                future.add_done_callback(
                    lambda f: logging.debug(
                        f"Wrote background timeslide {f.result()}"
                    )
                )
                futures.append(future)

                # pack up polarizations in compatible format
                # with ml4gw project_raw_gw
                polarizations = {
                    "cross": torch.Tensor(waveforms[:, 0, :]),
                    "plus": torch.Tensor(waveforms[:, 1, :]),
                }

                logging.debug(
                    "Projecting and computing snrs for {} waveforms"
                    " on timeslide {}".format(len(waveforms), shift_str)
                )
                # project raw waveforms onto ifos to produce observed strain
                signals = compute_observed_strain(
                    torch.Tensor(parameters["dec"]),
                    torch.Tensor(parameters["psi"]),
                    torch.Tensor(parameters["ra"]),
                    tensors,
                    vertices,
                    sample_rate,
                    **polarizations,
                )

                # create psds from background timeseries
                # and pack up into tensors
                # compatible with ml4gw compute_ifo_snr
                df = 1 / (signals.shape[-1] / sample_rate)
                psds = []
                for ifo in ifos:
                    psd = background[ifo].psd(fftlength).interpolate(df)
                    psd = torch.tensor(psd.value, dtype=torch.float64)
                    psds.append(psd)
                psds = torch.stack(psds)

                snrs = compute_ifo_snr(
                    signals.type(torch.float64),
                    psds,
                    sample_rate,
                    highpass=highpass,
                )

                logging.debug(
                    "Completed projection of {} waveforms and snr computation "
                    "timeslide {} ".format(len(waveforms), shift_str)
                )
                for i, ifo in enumerate(ifos):
                    parameters[f"{ifo}_snr"] = snrs[:, i]

                # inject the projected waveforms into the background
                logging.debug(
                    "Beginning injection of {} waveforms "
                    "on timeslide {}".format(len(waveforms), shift_str)
                )
                signals = signals.numpy()

                # inject projected waveforms into background data
                injected_data = {}
                for i, ifo in enumerate(ifos):
                    injected_data[ifo] = inject_waveforms(
                        (times, background_data[ifo]),
                        signals[:, i, :],
                        parameters["geocent_time"],
                    )

                logging.debug(
                    "completed injection of {} waveforms on "
                    "timeslide {}".format(len(waveforms), shift_str)
                )

                # submit this injected data as a write job to the process pool
                future = process_pool.submit(
                    h5.write_timeseries,
                    injection_ts.path,
                    prefix="inj",
                    t=times,
                    **injected_data,
                )
                future.add_done_callback(
                    lambda f: logging.debug(
                        f"Wrote injected timeslide {f.result()}"
                    )
                )
                futures.append(future)

                with h5py.File(injection_ts.path / "params.h5", "a") as f:
                    for k, v in parameters.items():
                        if k not in f:
                            max_shape = (None,)
                            if v.ndim > 1:
                                max_shape += v.shape[1:]
                            f.create_dataset(k, data=v, maxshape=max_shape)
                        else:
                            dataset = f[k]
                            dataset.resize(len(dataset) + len(v), axis=0)
                            dataset[-len(v) :] = v

            # don't move on until we've finished writing
            # everything so that we don't accidentally
            # go out of memory. TODO: this is still possible
            # if one segment is so big that all its slides
            # go OOM. How to monitor and prevent this?
            wait(futures, return_when=FIRST_EXCEPTION)


if __name__ == "__main__":
    main()
