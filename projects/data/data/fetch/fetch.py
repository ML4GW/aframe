from typing import List

from gwpy.timeseries import TimeSeries, TimeSeriesDict

# channel names that signal to fetch open data
OPEN_DATA_CHANNELS = ["H1", "L1", "V1"]


def _fetch_open_data(
    ifos: List[str], start: float, end: float, **kwargs
) -> TimeSeriesDict:
    ts_dict = TimeSeriesDict()
    for ifo in ifos:
        ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, **kwargs)
    return ts_dict


def fetch(
    start: float,
    end: float,
    channels: list[str],
    sample_rate: float,
    nproc: int = 3,
    verbose: bool = True,
    allow_tape: bool = True,
) -> TimeSeriesDict:
    """
    Simple wrapper to annotate and simplify
    the kwargs so that jsonargparse can build
    a parser out of them.
    """
    open_data_channels = list(
        filter(lambda x: x in OPEN_DATA_CHANNELS, channels)
    )
    channels = list(filter(lambda x: x not in OPEN_DATA_CHANNELS, channels))

    # fetch data from nds2
    data = TimeSeriesDict()
    if channels:
        for channel in channels:
            ifo = channel.split(":")[0]
            data[ifo] = TimeSeries.get(
                channel,
                start=start,
                end=end,
                verbose=verbose,
                nproc=nproc,
                allow_tape=allow_tape,
            )

    # fetch open data channels and combine
    if open_data_channels:
        open_data_ts_dict = _fetch_open_data(
            open_data_channels,
            start=start,
            end=end,
            verbose=verbose,
            nproc=nproc,
        )
        data.update(open_data_ts_dict)

    data = data.resample(sample_rate)
    return data
