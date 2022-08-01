import logging
from pathlib import Path
from typing import Iterable, Optional

import gwdatafind
import numpy as np
from gwpy.segments import (
    DataQualityDict,
    Segment,
    SegmentList,
    SegmentListDict,
)
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from bbhnet.injection import inject_signals_into_timeslide
from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor
from hermes.typeo import typeo


@typeo
def main(
    start: int,
    stop: int,
    logdir: Path,
    datadir: Path,
    prior_file: str,
    spacing: float,
    jitter: float,
    buffer_: float,
    n_slides: int,
    shifts: Iterable[float],
    ifos: Iterable[str],
    file_length: int,
    fmin: float,
    sample_rate: float,
    frame_type: str,
    channel: str,
    waveform_duration: float = 8,
    reference_frequency: float = 20,
    waveform_approximant: str = "IMRPhenomPv2",
    fftlength: float = 2,
    state_flag: Optional[str] = None,
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
        fmin: min frequency for highpass filter, used for simulating
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
    configure_logging(logdir / "timeslide_injections.log")

    # query and read all necessary data up front
    logging.info(f"Querying data from {start} to {stop}")
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
            files, channel=f"{ifo}:{channel}", start=start, end=stop
        )

    data.resample(sample_rate)

    # if state_flag is passed,
    # query segments for each ifo.
    # a certificate is needed for this
    segments = SegmentListDict()

    logging.info("Querying segments")
    if state_flag:
        # query science segments
        segments = DataQualityDict.query_dqsegdb(
            [f"{ifo}:{state_flag}" for ifo in ifos],
            start,
            stop,
        )
        # convert DQ dict to SegmentList Dict
        segments = SegmentListDict(
            {key: segments[key].active for key in segments.keys()}
        )

        # intersect segments
        intersection = segments.intersection(segments.keys())

    else:
        # not considering segments so
        # make intersection from start to stop
        intersection = SegmentList([Segment(start, stop)])

    # create list of timeslides
    # for each ifo
    timeslides = np.column_stack(
        [
            np.linspace(0, shift * (n_slides - 1), num=n_slides)
            for shift in shifts
        ]
    )

    for shifts in timeslides:

        # TODO: might be overly complex naming,
        # but wanted to attempt to generalize to multi ifo
        root = datadir / f"dt-{'-'.join(map(str,shifts))}"

        # make root and timeslide directories
        root.mkdir(exist_ok=True, parents=True)

        # create TimeSlide object for injection
        # this will create the directories if
        # they don't exist
        injection_ts = TimeSlide.create(root=root, field="injection")

        # create TimeSlide object for raw data
        # this will create the directories
        # if they don't exist
        raw_ts = TimeSlide.create(root=root, field="background")

        for segment in intersection:
            segment_start, segment_stop = segment
            segment_start, segment_stop = float(segment_start), float(
                segment_stop
            )

            segment_length = segment_stop - segment_start
            if segment_length < (n_slides * max(shifts)):
                logging.warning(
                    "Performing a circular timeshift on a segment shorter in"
                    " length then the longest timeshift: some timeslides will"
                    " be duplicates"
                )

            shifted_data = {}
            for shift, ifo in enumerate(ifos):
                # get data for this segment
                segment_data = (
                    data[ifo].crop(segment_start, segment_stop).value
                )
                times = data[ifo].crop(segment_start, segment_stop).times.value

                # roll timeseries by timeshift for ifo
                shifted_data[ifo] = np.roll(
                    segment_data, int(np.round(shift * sample_rate))
                )

            # write timeseries
            h5.write_timeseries(
                raw_ts.path, prefix="raw", t=times, **shifted_data
            )

        # update segments in TimeSlide

        raw_ts.update()

        # create process and thread pools
        thread_ex = AsyncExecutor(4, thread=True)
        process_ex = AsyncExecutor(4, thread=False)

        # now inject signals into raw files;
        # this function automatically writes h5 files to TimeSlide
        # for injected data
        with process_ex, thread_ex:
            inject_signals_into_timeslide(
                process_ex,
                thread_ex,
                raw_ts,
                injection_ts,
                ifos,
                prior_file,
                spacing,
                jitter,
                sample_rate,
                file_length,
                fmin,
                waveform_duration,
                reference_frequency,
                waveform_approximant,
                buffer_,
            )

    return datadir


if __name__ == "__main__":
    main()
