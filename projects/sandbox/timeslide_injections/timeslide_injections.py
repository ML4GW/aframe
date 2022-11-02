import itertools
import logging
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

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


@dataclass
class Shift:
    ifos: List[str]
    shifts: Iterable[float]

    def __post_init__(self):
        self.shifts = [float(i) for i in self.shifts]
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self.ifos):
            raise StopIteration

        ifo, shift = self.ifos[self._i], self.shifts[self._i]
        self._i += 1
        return ifo, shift

    def __str__(self):
        return "-".join([f"{i[0]}{j}" for i, j in zip(self.ifos, self.shifts)])


def make_shifts(
    ifos: Iterable[str], shifts: Iterable[float], n_slides: int
) -> List[Shift]:
    ranges = [range(n_slides) for i in shifts if i]
    shift_objs = []
    for rng in itertools.product(*ranges):
        it = iter(rng)
        shift = []
        for i in shifts:
            shift.append(0 if i == 0 else next(it) * i)
        shift = Shift(ifos, shift)
        shift_objs.append(shift)

    return shift_objs


def submit_write(
    pool: AsyncExecutor, ts: TimeSlide, t: np.ndarray, **fields: np.ndarray
) -> Future:
    ts_type = ts.path.name
    if ts_type == "background":
        prefix = "raw"
    else:
        prefix = "inj"

    future = pool.submit(
        h5.write_timeseries,
        ts.path,
        prefix=prefix,
        t=t,
        **fields,
    )

    future.add_done_callback(
        lambda f: logging.debug(f"Wrote background {ts_type} {f.result()}")
    )
    return future


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
    """
    Cheap wrapper around waveform generation for async
    execution so that we can know which parameters these
    waveforms correspond to.
    """

    signals = generate_gw(
        parameters,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
    )
    return signals, parameters


def intify(x: float):
    return int(x) if int(x) == x else x


def check_segment(
    shifts: List[Shift],
    datadir: Path,
    segment_start: float,
    dur: float,
    min_segment_length: Optional[float] = None,
    force_generation: bool = False,
):
    # first check if we'll even have enough data for
    # this segment to be worth working with
    if min_segment_length is not None and dur < min_segment_length:
        return None

    segment_start = intify(segment_start)
    dur = intify(dur)

    # then check if _all_ data for this segment
    # exists in each shift separately
    fields, prefixes = ["background", "injection"], ["raw", "inj"]
    segment_shifts = []
    for shift in shifts:
        for field, prefix in zip(fields, prefixes):
            dirname = datadir / f"dt-{shift}" / field
            fname = f"{prefix}_{segment_start}-{dur}.hdf5"
            if not (dirname / fname).exists() or force_generation:
                # we don't have data for this segment at this
                # shift value, so we'll need to create it
                segment_shifts.append(shift)
                break

    return segment_shifts


@scriptify
def main(
    start: int,
    stop: int,
    logdir: Path,
    datadir: Path,
    prior: Callable,
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
    force_generation: bool = False,
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
        prior: a prior function defined in prior.py script in the injection lib
        spacing: spacing between consecutive injections
        n_slides: number of timeslides
        shifts:
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

    # record some properties of our shifts then
    # convert them to more convenient Shift objects
    max_shift = max(shifts) * n_slides
    shifts = make_shifts(ifos, shifts, n_slides)

    # grab some parameters we'll need for waveform generation
    stride = 1 / sample_rate
    priors = prior()
    tensors, vertices = get_ifo_geometry(*ifos)

    # set up some pools for doing our data IO/injection
    with AsyncExecutor(4, thread=False) as pool:
        for segment_start, segment_stop in intersection:
            dur = segment_stop - segment_start - max_shift
            seg_str = f"{segment_start}-{segment_stop}"

            segment_shifts = check_segment(
                shifts,
                datadir,
                segment_start,
                dur,
                min_segment_length,
                force_generation,
            )
            if segment_shifts is None:
                logging.info(f"Segment {seg_str} too short, skipping")
                continue
            elif len(segment_shifts) == 0:
                logging.info(
                    f"All data for segment {seg_str} already exists, skipping"
                )
                continue
            num_shifts = len(segment_shifts)

            # generate timestamps of trigger times for each waveform.
            # Jitter will get added in `get_params`.
            # take off min segment length from end of segment stop
            # since we end up cutting some data off to make sure each timeslide
            # has the same length of data
            signal_start = segment_start + buffer_
            signal_stop = segment_stop - buffer_ - max_shift
            signal_times = np.arange(signal_start, signal_stop, spacing)

            # sample dictionary of params for each timeslide
            num_signals = num_shifts * len(signal_times)
            logging.debug(f"Sampling {num_signals} waveform parameters")
            parameters = get_params(
                priors, num_shifts, signal_times, jitter, waveform_duration
            )

            # generate each timeslide's waveforms in parallel
            logging.debug("Launching waveform generation in parallel")
            waveform_it = pool.imap(
                generate_waveforms,
                parameters,
                minimum_frequency=minimum_frequency,
                reference_frequency=reference_frequency,
                sample_rate=sample_rate,
                waveform_duration=waveform_duration,
                waveform_approximant=waveform_approximant,
            )

            # begin the download of data in a separate thread
            logging.debug(f"Beginning download of segment {seg_str}")
            background = download_data(
                ifos,
                frame_type,
                channel,
                sample_rate,
                segment_start,
                segment_stop,
            )
            logging.debug(f"Completed download of segment {seg_str}")

            # set up array of times for all shifts
            t = np.arange(segment_start, segment_start + dur, stride)
            futures = []
            it = zip(waveform_it, segment_shifts)
            for (waveforms, parameters), shift in it:
                logging.debug(
                    "Creating timeslide for segment {} "
                    "with shifts {}".format(seg_str, shift)
                )

                # 1. start by creating all the directories we'll need
                root = datadir / f"dt-{shift}"
                root.mkdir(exist_ok=True, parents=True)

                raw_ts = TimeSlide.create(root=root, field="background")
                injection_ts = TimeSlide.create(root=root, field="injection")

                # 2. Then create the appropriate shifts for each
                # interferometer and save them to their raw
                # directory

                # time array is always relative to first shift value
                times = t + shift.shifts[0]
                background_data = {}
                for ifo, shift_val in shift:
                    start = segment_start + shift_val
                    bckgrd = background[ifo].crop(start, start + dur)
                    background_data[ifo] = bckgrd.value

                future = submit_write(pool, raw_ts, t, **background_data)
                futures.append(future)

                # 3. Now project the waveforms for this timeshift
                # to the indicated interferometers

                # pack up polarizations in compatible format
                # with ml4gw project_raw_gw
                polarizations = {
                    "cross": torch.Tensor(waveforms[:, 0, :]),
                    "plus": torch.Tensor(waveforms[:, 1, :]),
                }

                logging.debug(
                    "Projecting and computing snrs for {} waveforms"
                    " on timeslide {}".format(len(waveforms), shift)
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

                # 4. Compute the SNRs of the injected waveforms
                # to record as metadata with the injections

                # create psds from background timeseries
                # and pack up into tensors compatible
                # with ml4gw compute_ifo_snr
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
                snrs = snrs.numpy()

                logging.debug(
                    "Completed projection of {} waveforms and snr computation "
                    "timeslide {} ".format(len(waveforms), shift)
                )
                for i, ifo in enumerate(ifos):
                    parameters[f"{ifo}_snr"] = snrs[:, i]

                # 5. Inject the projected waveforms into the background
                logging.debug(
                    "Beginning injection of {} waveforms "
                    "on timeslide {}".format(len(waveforms), shift)
                )
                signals = signals.numpy()
                injected_data = {}
                for i, ifo in enumerate(ifos):
                    injected_data[ifo] = inject_waveforms(
                        (times, background_data[ifo]),
                        signals[:, i, :],
                        parameters["geocent_time"],
                    )

                logging.debug(
                    "completed injection of {} waveforms on "
                    "timeslide {}".format(len(waveforms), shift)
                )

                # 6. Write the injected data for this shift to
                # the corresponding injection directory
                future = submit_write(pool, injection_ts, t, **injected_data)
                futures.append(future)

                # 7. Write the injection parameters to the injection
                # directory as metadata for downstream processes
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
