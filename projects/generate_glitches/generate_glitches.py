import logging
from pathlib import Path
from typing import Optional

import gwdatafind
import h5py
import numpy as np
from gwpy.segments import Segment, SegmentList
from gwpy.timeseries import TimeSeries
from hermes.typeo import typeo
from tqdm import tqdm

"""
Script to generate a dataset of glitches from omicron triggers.

For information on how the omicron triggers were generated see:

/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/12566/H1L1_1256665618_100000
/runfiles/omicron_params_H1.txt
on CIT cluster for an example omicron parameter file.

The code used to generate the omicron runs is based on the oLIB algorithm.
Email emarx@mit.edu for any questions or concerns
"""


def veto(times: list, segmentlist: SegmentList):

    """
    Remove events from a list of times based on a segmentlist
    A time ``t`` will be vetoed if ``start <= t <= end`` for any veto
    segment in the list.

    Arguments:
    - times: the times of event triggers to veto
    - segmentlist: the list of veto segments to use

    Returns:
    - keep_bools: list of booleans; True for the triggers to keep
    """

    # find args that sort times and create sorted times array
    sorted_args = np.argsort(times)
    sorted_times = times[sorted_args]

    # initiate array of args to keep;
    # refers to original args of unsorted times array;
    # begin with all args being kept

    keep_bools = np.ones(times.shape[0], dtype=bool)

    # initiate loop variables; extract first segment
    j = 0
    a, b = segmentlist[j]
    i = 0

    while i < sorted_times.size:
        t = sorted_times[i]

        # if before start, not in vetoed segment; move to next trigger now
        if t < a:

            # original arg is the ith sorted arg
            i += 1
            continue

        # if after end, find the next segment and check this trigger again
        if t > b:
            j += 1
            try:
                a, b = segmentlist[j]
                continue
            except IndexError:
                break

        # otherwise it must be in veto segment; move on to next trigger
        original_arg = sorted_args[i]
        keep_bools[original_arg] = False
        i += 1

    return keep_bools


def generate_glitch_dataset(
    ifo: str,
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    sample_rate: float,
    channel: str,
    frame_type: str,
    trig_file: str,
    vetoes: SegmentList = None,
):

    """
    Generates a list of omicron trigger times that satisfy snr threshold

    Arguments:
    - ifo: ifo to generate glitch triggers for
    - snr_thresh: snr threshold above which to keep as glitch
    - start: start gpstime
    - stop: stop gpstime
    - window: half window around trigger time to query data for
    - sample_rate: sampling frequency
    - channel: channel name used to read data
    - frame_type: frame type for data discovery w/ gwdatafind
    - trig_file: txt file output from omicron triggers
            (first column is gps times, 3rd column is snrs)
    - vetoes: SegmentList object of times to ignore
    """
    glitches = []
    snrs = []

    # snr and time columns in omicron file
    snr_col = 2
    time_col = 0

    # load in triggers
    triggers = np.loadtxt(trig_file)

    # restrict triggers to within gps start and stop times
    # and apply snr threshold
    times = triggers[:, time_col]
    mask = (times > start) & (times < stop)
    mask &= triggers[:, snr_col] > snr_thresh
    triggers = triggers[mask]

    # if passed, apply vetos
    if vetoes is not None:
        keep_bools = veto(triggers[:, time_col], vetoes)
        triggers = triggers[keep_bools]

    # re-set 'start' and 'stop' so we aren't querying unnecessary data
    start = np.min(triggers[:, time_col]) - 2 * window
    stop = np.max(triggers[:, time_col]) + 2 * window

    logging.info(
        f"Querying {stop - start} seconds of data for {len(triggers)} triggers"
    )

    # use gwdatafind to create frame cache
    frames = gwdatafind.find_urls(
        site=ifo.strip("1"),
        frametype=f"{ifo}_{frame_type}",
        gpsstart=int(start),
        gpsend=int(stop),
        urltype="file",
        on_gaps="ignore",
    )

    # read frames all at once, then crop
    # this time series for each trigger.
    # Although its initially slower to read all the frames at once,
    # I think this is overall faster than querying a 4096 frame for
    # every trigger.

    ts = TimeSeries.read(
        frames, channel=f"{ifo}:{channel}", start=start, end=stop, pad=0
    )

    # for each trigger
    for trigger in tqdm(triggers):
        time = trigger[time_col]

        try:
            glitch_ts = ts.crop(time - window, time + window)

            glitch_ts = glitch_ts.resample(sample_rate)

            snrs.append(trigger[snr_col])
            glitches.append(glitch_ts)

        except ValueError:
            logging.warning(f"Data not available for trigger at time: {time}")
            continue

    glitches = np.array(glitches)
    snrs = np.array(snrs)
    return glitches, snrs


@typeo
def main(
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    omicron_dir: Path,
    out_dir: Path,
    channel: str,
    frame_type: str,
    sample_rate: float = 4096,
    H1_veto_file: Optional[str] = None,
    L1_veto_file: Optional[str] = None,
):

    """Simulates a set of glitches for both
        H1 and L1 that can be added to background

    Arguments:

    - snr_thresh: snr threshold above which to keep as glitch
    - start: start gpstime
    - stop: stop gpstime
    - window: half window around trigger time to query data for
    - sample_rate: sampling frequency
    - out_dir: output directory to which signals will be written
    - omicron_dir: base directory of omicron triggers
            (see /home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/)
    - channel: channel name used to read data
    - frame_type: frame type for data discovery w/ gwdatafind
    - H1_veto_file: path to file containing vetoes for H1
    - L1_veto_file: path to file containing vetoes for L1
    """

    # create logging file in model_dir
    logging.basicConfig(
        filename=out_dir / "log.log",
        format="%(message)s",
        filemode="w",
        level=logging.INFO,
    )

    # if passed, load in H1 vetoes and convert to gwpy SegmentList object
    if H1_veto_file is not None:

        logging.info("Applying vetoes to H1 times")
        logging.info(f"H1 veto file: {H1_veto_file}")

        # load in H1 vetoes
        H1_vetoes = np.loadtxt(H1_veto_file)

        # convert arrays to gwpy Segment objects
        H1_vetoes = [Segment(seg[0], seg[1]) for seg in H1_vetoes]

        # create SegmentList object
        H1_vetoes = SegmentList(H1_vetoes).coalesce()
    else:
        H1_vetoes = None

    if L1_veto_file is not None:
        logging.info("Applying vetoes to L1 times")
        logging.info(f"L1 veto file: {L1_veto_file}")

        L1_vetoes = np.loadtxt(L1_veto_file)
        L1_vetoes = [Segment(seg[0], seg[1]) for seg in L1_vetoes]
        L1_vetoes = SegmentList(L1_vetoes).coalesce()
    else:
        L1_vetoes = None

    # omicron triggers are split up by directories
    # into segments of 10^5 seconds
    # get paths for relevant directories
    # based on start and stop gpstimes passed by user

    # TODO: I *Think* there is a pyomicron package
    # that can create omicron dags. Might be useful
    # to generalize this script to use pyomicron to produce
    # omicron triggers for arbitrary stretches of data,
    # channels, parameters etc..

    gps_day_start = start // 100000
    gps_day_end = stop // 100000
    all_gps_days = np.arange(int(gps_day_start), int(gps_day_end) + 1, 1)

    H1_glitches = []
    L1_glitches = []

    H1_snrs = []
    L1_snrs = []

    # loop over gps days
    for day in all_gps_days:

        # get path for this gps day
        omicron_day_path = omicron_dir / day
        trigger_path = omicron_day_path.glob(
            "*" / Path("PostProc/unclustered/")
        )

        # the path to the omicron triggers
        H1_trig_file = trigger_path / Path("triggers_unclustered_H1.txt")
        L1_trig_file = trigger_path / Path("triggers_unclustered_L1.txt")

        H1_day_glitches, H1_day_snrs = generate_glitch_dataset(
            "H1",
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            channel,
            frame_type,
            H1_trig_file,
            vetoes=H1_vetoes,
        )

        L1_day_glitches, L1_day_snrs = generate_glitch_dataset(
            "L1",
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            channel,
            frame_type,
            L1_trig_file,
            vetoes=L1_vetoes,
        )

        # concat
        H1_glitches.append(H1_day_glitches)
        L1_glitches.append(L1_day_glitches)

        H1_snrs.append(H1_day_snrs)
        L1_snrs.append(L1_day_snrs)

    glitch_file = out_dir / Path("glitches.h5")

    with h5py.File(glitch_file, "w") as f:
        f.create_dataset("H1_glitches", data=H1_glitches)
        f.create_dataset("H1_snrs", data=H1_snrs)

        f.create_dataset("L1_glitches", data=L1_glitches)
        f.create_dataset("L1_snrs", data=L1_snrs)

    return glitch_file


if __name__ == "__main__":
    main()
