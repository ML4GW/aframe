import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from pathlib import Path
import pandas as pd

IFOS = ["H1", "L1", "V1"]


def aframe_response_plot(event: Path, plotsdir: Path, gpstime: float) -> None:
    """
    Create a plot of Aframe's response

    Args:
        event: Path to the event directory
        plotsdir: Directory to save the plots
        gpstime: GPS time of the event
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
    plt.title("Aframe Response")
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


def whitened_plot(
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
        qplot = ts.q_transform(
            whiten=False,
            gps=gpstime,
            logf=True,
            frange=(online_args["amplfi_highpass"], np.inf),
        ).plot(epoch=gpstime)
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
    plt.hist(df["latency"], bins=30, alpha=0.7)
    plt.xlabel("Aframe latency (s)")
    plt.ylabel("Count")
    plt.savefig(plotsdir / "aframe_latency.png", dpi=150)
    plt.close()
