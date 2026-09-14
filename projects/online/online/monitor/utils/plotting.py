import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.time import tconvert
from gwpy.timeseries import TimeSeries
from pathlib import Path
import pandas as pd
from scipy.stats import poisson
import warnings
import logging

from datetime import datetime, timedelta, timezone

from online.monitor.utils.segments import compute_duty_cycle
from online.utils.segments import PipelineState
from online.utils.timing import gps_now

IFOS = ["H1", "L1", "V1"]
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 365 * SECONDS_PER_DAY

STATE_LABELS = {
    PipelineState.ANALYZING: "Analyzing",
    PipelineState.WARMUP: "Filter warm-up",
    PipelineState.STARTUP: "Starting up",
    PipelineState.SEARCH_DOWN: "Search down",
    PipelineState.NOT_READY: "Detectors not ready",
    PipelineState.MISSING_DATA: "Missing data",
}

_CYCLE = plt.style.library["tableau-colorblind10"]["axes.prop_cycle"]
CB_COLORS = _CYCLE.by_key()["color"]
STATE_COLORS = {
    PipelineState.ANALYZING: CB_COLORS[0],
    PipelineState.WARMUP: CB_COLORS[1],
    PipelineState.STARTUP: CB_COLORS[8],
    PipelineState.SEARCH_DOWN: CB_COLORS[5],
    PipelineState.NOT_READY: CB_COLORS[2],
    PipelineState.MISSING_DATA: CB_COLORS[3],
}

for _table, _name in ((STATE_LABELS, "label"), (STATE_COLORS, "color")):
    _missing = [state for state in PipelineState if state not in _table]
    if _missing:
        raise RuntimeError(
            f"No {_name} defined for pipeline state(s): " + ", ".join(_missing)
        )

logger = logging.getLogger("monitor-plotting")


def aframe_response_plot(
    event: Path, plotsdir: Path, gpstime: float, far: float
) -> None:
    """
    Create a plot of Aframe's response

    Args:
        event: Path to the event directory
        plotsdir: Directory to save the plots
        gpstime: GPS time of the event
        far: False alarm rate of the event
    """
    with h5py.File(event / "output.hdf5", "r") as f:
        time = f["time"][:]
        output = f["output"][:]
        integrate = f["integrated"][:]

    plt.figure(figsize=(10, 6))
    plt.plot(time, output, label="Raw output")
    plt.plot(time, integrate, label="Integrated output")
    plt.axvline(gpstime, color="red", linestyle="--", label="Event time")
    plt.xlabel("GPS time")
    plt.ylabel("Detection statistic")
    plt.legend()
    plt.title(f"Aframe Response (FAR: {far:.2e} Hz)")
    plt.savefig(plotsdir / "aframe_response.png", dpi=150)
    plt.close()


def asd_plot(event: Path, plotsdir: Path) -> None:
    """
    Create a plot of the amplitude spectral densities

    Args:
        event: Path to the event directory
        plotsdir: Directory to save the plots
    """
    asds = np.load(event / "asd.npy")[0]
    freqs = asds[0]
    asds = asds[1:]

    plt.figure(figsize=(10, 6))
    for i, ifo in enumerate(IFOS[: len(asds)]):
        plt.plot(freqs, asds[i], label=ifo)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("ASD (strain/Hz^0.5)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Amplitude Spectral Densities")
    plt.savefig(plotsdir / "asds.png", dpi=150)
    plt.close()


def q_plots(
    event: Path, plotsdir: Path, gpstime: float, online_args: dict
) -> None:
    """
    Create Q-plots of the whitened AMPLFI data

    Args:
        event: Path to the event directory
        plotsdir: Directory to save the plots
        gpstime: GPS time of the event
        online_args: Dictionary containing online configuration parameters
    """
    whitened = np.load(event / "amplfi_whitened.npy")[0]
    sample_rate = online_args["sample_rate"]
    t0 = gpstime - online_args["event_position"]
    for i, ifo in enumerate(IFOS[: len(whitened)]):
        ts = TimeSeries(whitened[i], sample_rate=sample_rate, t0=t0)
        try:
            qplot = ts.q_transform(
                whiten=False,
                gps=gpstime,
                logf=True,
                frange=(online_args["amplfi_highpass"], np.inf),
            ).plot(epoch=gpstime)
        except ValueError:
            logger.info(f"Failed to create Q-plot for {ifo}")
            continue
        ax = qplot.gca()
        ax.set_yscale("log")
        qplot.savefig(plotsdir / f"{ifo}_qtransform.png", dpi=150)
        plt.close()


def latency_plot(plotsdir: Path, df: pd.DataFrame) -> None:
    """
    Create a histogram of the Aframe latencies

    Args:
        plotsdir: Directory to save the plots
        df: DataFrame containing event data
    """
    plt.figure(figsize=(10, 6))
    latency = df["aframe latency"].dropna()
    median = latency.median()
    ninetieth_percentile = latency.quantile(0.9)
    bins = np.logspace(
        np.log10(latency.min()), np.log10(latency.max()), num=30
    )
    plt.hist(latency, bins=bins, alpha=0.7)
    plt.axvline(
        median, color="red", linestyle="--", label=f"Median: {median:.2f} s"
    )
    plt.axvline(
        ninetieth_percentile,
        color="k",
        linestyle="--",
        label=f"90th Percentile: {ninetieth_percentile:.2f} s",
    )
    plt.xscale("log")
    plt.xlabel("Aframe latency (s)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(plotsdir / "aframe_latency.png", dpi=150)
    plt.close()


def ifar_plot(plotsdir: Path, df: pd.DataFrame, tb: float) -> None:
    """
    Create a plot of the inverse false alarm rate (iFAR) vs the
    cumulative event distribution.

    Args:
        plotsdir: Directory to save the plots
        df: DataFrame containing event data
        tb: Background livetime in seconds
    """
    ifars = np.sort(1 / df["far"]) / SECONDS_PER_YEAR
    counts = np.arange(1, len(ifars) + 1)[::-1]

    y_vals = np.logspace(np.log10(max(counts) * 1.2), -3, 1000)
    x_pred = tb / y_vals / SECONDS_PER_YEAR

    sigma_alphas = (0.4, 0.25, 0.1)
    sig_bands = []
    for fap in (0.682689, 0.954499, 0.997300):
        band = np.array(poisson.interval(fap, y_vals))
        band[0] = np.maximum(band[0] - 0.5, 0)
        band[1] += 0.5
        sig_bands.append(band)

    plt.figure(figsize=(10, 6))
    plt.plot(x_pred, y_vals, color="steelblue", lw=2, label="Predicted")
    plt.step(
        ifars,
        counts,
        where="post",
        lw=2,
        marker="o",
        ms=3,
        color="#f29e4c",
        label="Zero‑lag",
    )
    labels = [r"1,2,3 $\,\sigma$"] + [None, None]
    for band, alpha, lab in zip(
        sig_bands[::-1], sigma_alphas[::-1], labels, strict=True
    ):
        plt.fill_between(
            x_pred, band[0], band[1], color="steelblue", alpha=alpha, label=lab
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(ifars.min() * 0.1, ifars.max() * 10.2)
    plt.ylim(0.8, counts.max() * 1.2)
    plt.xlabel("Inverse false alarm rate (years)")
    plt.ylabel("Count")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.savefig(plotsdir / "ifar_plot.png", dpi=150)
    plt.close()


def duty_cycle_plots(plotsdir: Path, segments: pd.DataFrame) -> None:
    """
    Create plots of how the search spent its time.

    Args:
        plotsdir: Directory to save the plots.
        segments: Segment DataFrame from `load_segments`.
    """
    _duty_cycle_timeline(plotsdir, segments)
    _duty_cycle_trend(plotsdir, segments)


def _duty_cycle_timeline(plotsdir: Path, segments: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    origin = segments["start"].min()
    starts = (segments["start"] - origin) / 3600
    stops = (segments["stop"] - origin) / 3600
    for row, state in enumerate(PipelineState):
        in_state = segments["state"] == state
        spans = [
            (start, stop - start)
            for start, stop in zip(
                starts[in_state], stops[in_state], strict=True
            )
        ]
        ax.broken_barh(
            spans, (row - 0.4, 0.8), facecolors=STATE_COLORS[state], zorder=2
        )

    ax.set_yticks(range(len(PipelineState)))
    ax.set_yticklabels([STATE_LABELS[s] for s in PipelineState])
    ax.set_xlabel(f"Hours since {tconvert(origin).strftime('%Y-%m-%d %H:%M')}")
    ax.set_xlim(0, max(stops.max(), 1 / 60))
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    plt.savefig(
        plotsdir / "duty_cycle_timeline.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def _duty_cycle_trend(plotsdir: Path, segments: pd.DataFrame) -> None:
    """
    Hourly duty cycle over the past week. Hours where the detectors
    gave us nothing to analyze are left as gaps in the line.
    """
    now = gps_now()
    start = max(now - 7 * SECONDS_PER_DAY, segments["start"].min())
    edges = np.arange(start, now, 3600)

    plt.figure(figsize=(10, 6))
    if len(edges) > 1:
        fractions = [
            compute_duty_cycle(segments, start=low, end=high)["duty_cycle"]
            for low, high in zip(edges[:-1], edges[1:], strict=True)
        ]
        plt.plot(
            (edges[:-1] - now) / 3600,
            [np.nan if f is None else 100 * f for f in fractions],
            marker="o",
            ms=3,
            color=STATE_COLORS[PipelineState.ANALYZING],
        )

    plt.xlabel("Hours ago")
    plt.ylabel("Search duty cycle [%]")
    plt.ylim(-2, 102)
    plt.grid(True, ls="--", alpha=0.4)
    plt.savefig(
        plotsdir / "duty_cycle_trend.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def event_rate_plots(plotsdir: Path, df: pd.DataFrame) -> None:
    """
    Create event rate plots over various time scales.

    Args:
        plotsdir: Directory to save the plots.
        df: DataFrame containing event data.
    """
    plt.figure(figsize=(10, 6))
    df.set_index("datetime", inplace=True)
    df = df.tz_localize(timezone.utc)

    current_time = datetime.now(timezone.utc)
    one_day_ago = current_time - timedelta(days=1)
    one_week_ago = current_time - timedelta(weeks=1)

    past_day = df[df.index >= one_day_ago]
    past_week = df[df.index >= one_week_ago]

    # Plotting raises warnings if there's not enough analysis time.
    # It doesn't harm anything, so catch them to make output cleaner.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        past_day = past_day.tz_convert(timezone.utc)
        past_week = past_week.tz_convert(timezone.utc)
        if len(past_day) > 0:
            past_day.resample("1h").size().plot()
        else:
            plt.figure()
        plt.xlabel("Time")
        plt.ylabel("Events per hour")
        plt.savefig(
            plotsdir / "event_rate_past_day.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        if len(past_week) > 0:
            past_week.resample("4h").size().plot()
        else:
            plt.figure()
        plt.xlabel("Time")
        plt.ylabel("Events per 4 hours")
        plt.savefig(
            plotsdir / "event_rate_past_week.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # If this plotting function is getting called, df will always have
        # at least one entry
        df.resample("1d").size().plot()
        plt.xlabel("Time")
        plt.ylabel("Events per day")
        plt.savefig(
            plotsdir / "event_rate_all_time.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
