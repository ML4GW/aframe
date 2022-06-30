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
from hermes.typeo import typeo


def circular_shift_segments(
    segments: SegmentList,
    shift: float,
    start: float,
    stop: float,
):
    """Takes a gwpy SegmentList object and performs a circular time shift.
    The convention we adopt is that a positive timeshift corresponds to
    moving the data forward in time.
    """

    if shift < 0:
        raise NotImplementedError(
            "circularly shifting segments is not"
            " yet implemented for negative shifts"
        )

    # shift segments by specified amount
    shifted_segments = segments.shift(shift)

    # create output of circularly shifted segments
    circular_shifted_segments = SegmentList([])

    # create full segment from start to stop
    # to use for deciding if part of a segment
    # needs to wrap around to the front
    full_segment = Segment([start, stop])

    for segment in shifted_segments:
        seg_start, seg_stop = segment

        # if segment is entirely between
        # start and stop just append
        if segment in full_segment:
            circular_shifted_segments.append(segment)

        # the entire segment got shifted
        # past the stop, so loop the segment around
        elif seg_start > stop:

            segment = segment.shift(start - stop)
            circular_shifted_segments.append(segment)

        # only a portion of the segment got shifted to front
        # so need to split up the segment
        elif seg_stop > stop:
            first_segment = Segment([seg_start, stop])
            second_segment = Segment([start, seg_stop - stop])
            circular_shifted_segments.extend([first_segment, second_segment])

    circular_shifted_segments = circular_shifted_segments.coalesce()
    return circular_shifted_segments


@typeo
def main(
    start: int,
    stop: int,
    outdir: Path,
    prior_file: str,
    spacing: float,
    buffer: float,
    n_slides: int,
    shifts: Iterable[float],
    ifos: Iterable[str],
    file_length: int,
    fmin: float,
    sample_rate: float,
    frame_type: str,
    channel: str,
    circular: bool = False,
    waveform_duration: float = 8,
    reference_frequency: float = 20,
    waveform_approximant: str = "IMRPhenomPv2",
    fftlength: float = 2,
    state_flag: Optional[str] = None,
):
    """Generates timeslides of background and background + injections.

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
        circular: flag for performing circular time shifts
        waveform_duration: length of injected waveforms
        reference_frequency: reference frequency for generating waveforms
        waveform_approximant: waveform model to inject
        fftlength: fftlength for calculating psd
        state_flag: name of segments to query from segment database
    """

    outdir.mkdir(parents=True, exist_ok=True)
    configure_logging(outdir / "timeslide_injections.log")

    # query and read all necessary data up front

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

    else:
        # make segment from start to stop
        segments = SegmentListDict()
        for ifo in ifos:
            segments[f"{ifo}:{state_flag}"] = SegmentList(
                [Segment(start, stop)]
            )

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
        root = outdir / f"dt-{'-'.join(map(str,shifts))}"

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

        # initiate segment intersection as full
        # segment from start, stop
        intersection = SegmentList([[start, stop]])

        # shift data
        # shift segments to 'mirror' data
        shifted_data = TimeSeriesDict()
        for shift, ifo in zip(shifts, ifos):

            # make a copy of segments
            # as some operations are done
            # in place
            segments_copy = segments.copy()

            # if circular timeshift
            if circular:
                shifted_segments = circular_shift_segments(
                    segments_copy[f"{ifo}:{state_flag}"],
                    shift,
                    start,
                    stop,
                )

                shifted_data = np.roll(
                    data[ifo].value, int(shift * sample_rate)
                )
                shifted_data[ifo] = TimeSeries(
                    shifted_data, dt=1 / sample_rate, t0=start
                )

            # global shift
            else:
                shifted_segments = segments_copy[f"{ifo}:{state_flag}"].shift(
                    shift
                )

                # perform time shift by manually shifting times;
                # subtracting the shift corresponds to moving
                # data forward in time
                shifted_times = data[ifo].times.value - shift
                shifted_data[ifo] = TimeSeries(
                    data[ifo].value, times=shifted_times
                )

            # calculate intersection of shifted segments
            intersection &= shifted_segments

        if len(intersection) == 0:
            logging.info(
                f"No intersecting segments found for {ifos}"
                " after time shifting by {shifts}"
            )
            continue

        for segment in intersection:
            segment_start, segment_stop = segment
            segment_start, segment_stop = float(segment_start), float(
                segment_stop
            )

            # write timeseries
            for t0 in np.arange(segment_start, segment_stop, file_length):

                tf = min(t0 + file_length, segment_stop, stop)
                raw_datasets = {}

                for ifo in ifos:
                    raw_datasets[ifo] = data[ifo].crop(t0, tf).value

                times = np.arange(t0, tf, 1 / sample_rate)

                h5.write_timeseries(
                    raw_ts.path, prefix="raw", t=times, **raw_datasets
                )

        # update segments in TimeSlide

        raw_ts.update()

        # now inject signals into raw files;
        # this function automatically writes h5 files to TimeSlide
        # for injected data
        inject_signals_into_timeslide(
            raw_ts,
            injection_ts,
            ifos,
            prior_file,
            spacing,
            sample_rate,
            file_length,
            fmin,
            waveform_duration,
            reference_frequency,
            waveform_approximant,
            buffer,
        )

    return outdir


if __name__ == "__main__":
    main()
