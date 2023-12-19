import logging

from gwpy.timeseries import TimeSeries, TimeSeriesDict


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

    logging.info(
        "Fetching {}s worth of data starting at GPS timestamp {}".format(
            end - start, start
        )
    )

    X = TimeSeriesDict()
    for channel in channels:
        ifo = channel.split(":")[0]
        logging.info(f"Fetching data for channel {channel}")

        X[ifo] = TimeSeries.get(
            channel,
            start,
            end,
            verbose=verbose,
            allow_tape=allow_tape,
            nproc=nproc,
        )

    logging.info(f"Data downloaded, resampling to {sample_rate}Hz")
    return X.resample(sample_rate)
