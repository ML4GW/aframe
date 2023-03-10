import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
from typeo import scriptify

from bbhnet.logging import configure_logging


@scriptify
def main(
    start: float,
    stop: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    frame_type: str,
    state_flag: str,
    minimum_length: float,
    output_file: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates background data for training BBHnet

    Args:
        start: start gpstime
        stop: stop gpstime
        ifos: which ifos to query data for
        outdir: where to store data
    """
    # make logdir dir
    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "generate_background.log", verbose)

    # check if output file already exists
    if output_file.exists() and not force_generation:
        # check the timestamp and verify that it
        # meets the conditions
        with h5py.File(output_file, "r") as f:
            missing_keys = [i not in f for i in ifos]
            if missing_keys:
                raise ValueError(
                    "Background file {} missing data from {}".format(
                        output_file, ", ".join(missing_keys)
                    )
                )

            t0 = f.attrs["t0"][()]
            length = f[ifos[0]][:] / sample_rate

        in_range = start <= t0 <= (stop - minimum_length)
        long_enough = length >= minimum_length
        if in_range and long_enough:
            logging.info(
                "Background data already exists and forced "
                "generation is off. Not generating background"
            )
            return
        else:
            raise ValueError(
                "Background file {} has t0 {} and length {}s, "
                "which isn't compatible with request of {}s "
                "segment between {} and {}".format(
                    output_file, t0, length, minimum_length, start, stop
                )
            )

    # make the output file's directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # query segments for each ifo
    # I think a certificate is needed for this
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        start,
        stop,
    )

    # create copy of first ifo segment list to start
    intersection = segments[f"{ifos[0]}:{state_flag}"].active.copy()

    # loop over ifos finding segment intersection
    for ifo in ifos:
        intersection &= segments[f"{ifo}:{state_flag}"].active

    # find first continuous segment of minimum length
    for seg_start, seg_stop in intersection:
        if (seg_stop - seg_start) >= minimum_length:
            break
    else:
        raise ValueError(
            "No segments of minimum length, not producing background"
        )

    logging.info(
        "Querying coincident, continuous segment "
        "from {} to {}".format(seg_start, seg_stop)
    )

    for ifo in ifos:

        # find frame files
        files = find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=start,
            gpsend=stop,
            urltype="file",
        )
        data = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=seg_start, end=seg_stop
        )

        # resample
        data = data.resample(sample_rate)

        if np.isnan(data).any():
            raise ValueError(
                f"The background for ifo {ifo} contains NaN values"
            )

        with h5py.File(output_file, "w") as f:
            for ifo in ifos:
                f.create_dataset(f"{ifo}", data=data)

            f.attrs["t0"] = seg_start

    return output_file
