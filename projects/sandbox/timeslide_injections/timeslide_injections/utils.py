import itertools
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bilby
import gwdatafind
import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from bbhnet.injection import generate_gw
from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide
from bbhnet.parallelize import AsyncExecutor


class Sampler:
    def __init__(
        self,
        priors: bilby.gw.prior.PriorDict,
        start: float,
        stop: float,
        buffer: float,
        max_shift: float,
        spacing: float,
        jitter: float,
    ) -> None:
        self.priors = priors
        self.signal_times = np.arange(
            start + buffer, stop - buffer - max_shift, spacing
        )
        self.jitter = jitter

    @property
    def num_signals(self):
        return len(self.signal_times)

    def __call__(self, waveform_duration: float):
        jit = np.random.uniform(
            -self.jitter, self.jitter, size=self.num_signals
        )
        times = self.signal_times + jit + waveform_duration / 2

        params = self.priors.sample(self.num_signals)
        params["geocent_time"] = times
        return params


@dataclass(frozen=True)
class WaveformGenerator:
    minimum_frequency: float
    reference_frequency: float
    sample_rate: float
    waveform_duration: float
    waveform_approximant: float

    def __call__(self, parameters):
        return generate_gw(
            parameters,
            self.minimum_frequency,
            self.reference_frequency,
            self.sample_rate,
            self.waveform_duration,
            self.waveform_approximant,
        )


def _generate_waveforms(sampler, generator):
    params = sampler(generator.waveform_duration)
    waveforms = generator(params)
    return waveforms, params


def waveform_iterator(
    pool: AsyncExecutor,
    sampler: Sampler,
    generator: WaveformGenerator,
    n_slides: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    future = pool.submit(_generate_waveforms, sampler, generator)
    for _ in range(n_slides - 1):
        waveforms, params = future.result()
        future = pool.submit(_generate_waveforms, sampler, generator)
        yield waveforms, params
    yield future.result()


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


def chunk_segments(segments: List[tuple], chunk_size: float):
    out_segments = []
    for segment in segments:
        start, stop = segment
        duration = stop - start
        if duration > chunk_size:
            num_segments = int((duration - 1) // chunk_size) + 1
            logging.info(f"Chunking segment into {num_segments} parts")
            for i in range(num_segments):
                end = min(start + (i + 1) * chunk_size, stop)
                seg = (start + i * chunk_size, end)
                out_segments.append(seg)
        else:
            out_segments.append(segment)
    return out_segments
