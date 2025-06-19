import json
import shutil
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import h5py

from .plotting import aframe_response_plot, asd_plot, whitened_plot

IFOS = ["H1", "L1", "V1"]


def get_plots(
    event: Path, eventdir: Path, gpstime: float, far: float, online_args: dict
) -> None:
    plotsdir = eventdir / "plots"
    plotsdir.mkdir(parents=True, exist_ok=True)
    for png in event.glob("*.png"):
        shutil.copy(png, plotsdir / png.name)

    aframe_response_plot(event, plotsdir, gpstime, far)
    asd_plot(event, plotsdir)
    whitened_plot(event, plotsdir, gpstime, online_args)


def update_dataframe(event: Path, outdir: Path) -> pd.DataFrame:
    df_file = outdir / "event_data.parquet"

    # Build event_dict from files
    url_path = event / "gracedb_url.txt"
    if url_path.exists():
        with url_path.open("r") as f:
            url = f.readline().strip()  # Strip newline
        event_dict = {
            "event": url.split("/")[-2],
            "url": url,
        }

    # Load main event metadata
    event_data_path = event / f"{event.stem}.json"
    if event_data_path.exists():
        with event_data_path.open("r") as f:
            metadata = json.load(f)
        event_dict.update(
            {
                "gpstime": metadata.get("gpstime"),
                "far": metadata.get("far"),
            }
        )

    # Load p_astro
    pastro_path = event / "aframe.p_astro.json"
    if pastro_path.exists():
        with pastro_path.open("r") as f:
            event_dict["p_bbh"] = json.load(f).get("BBH")

    # Load Aframe latency
    latency_path = event / "latency.log"
    if latency_path.exists():
        latencies = np.genfromtxt(latency_path, delimiter=",")
        event_dict["latency"] = float(latencies[1, -1])

    # Load posterior data
    posterior_path = event / "amplfi.posterior_samples.hdf5"
    if posterior_path.exists():
        with h5py.File(posterior_path, "r") as f:
            posterior = f["posterior_samples"][:]
            chirp_mass = posterior["chirp_mass"]
            mass_ratio = posterior["mass_ratio"]
            distance = posterior["luminosity_distance"]
        event_dict.update(
            {
                "chirp_mass_mean": np.mean(chirp_mass),
                "chirp_mass_median": np.median(chirp_mass),
                "mass_ratio_mean": np.mean(mass_ratio),
                "mass_ratio_median": np.median(mass_ratio),
                "distance_mean": np.mean(distance),
                "distance_median": np.median(distance),
            }
        )

    # Append to the existing DataFrame or create a new one
    if df_file.exists():
        prev_df = pd.read_parquet(df_file)
        df = pd.concat(
            [prev_df, pd.DataFrame([event_dict])], ignore_index=True
        )
    else:
        df = pd.DataFrame([event_dict])

    df.to_parquet(df_file, index=False)

    return df


def process_events(
    events: List[Path], outdir: Path, online_args: dict
) -> pd.DataFrame:
    for event in events:
        df = update_dataframe(event, outdir)
        with open(event / f"{event.stem}.json", "r") as f:
            data = json.load(f)
            gpstime = data.get("gpstime")
            far = data.get("far")
        eventdir = outdir / event.stem
        eventdir.mkdir(parents=True, exist_ok=True)
        try:
            get_plots(event, eventdir, gpstime, far, online_args)
        except FileNotFoundError:
            continue

    return df
