import json
import shutil
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from .plotting import aframe_response_plot, asd_plot, whitened_plot

IFOS = ["H1", "L1", "V1"]


def get_plots(
    event: Path, eventdir: Path, gpstime: float, online_args: dict
) -> None:
    plotsdir = eventdir / "plots"
    plotsdir.mkdir(parents=True, exist_ok=True)
    for png in event.glob("*.png"):
        shutil.copy(png, plotsdir / png.name)

    aframe_response_plot(event, plotsdir, gpstime)
    asd_plot(event, plotsdir)
    whitened_plot(event, plotsdir, gpstime, online_args)


def update_dataframe(event: Path, outdir: Path) -> pd.DataFrame:
    df_file = outdir / "event_data.parquet"

    # Build event_dict from files
    url_path = event / "gracedb_url.txt"
    with url_path.open("r") as f:
        url = f.readline().strip()  # Strip newline
    event_dict = {
        "event": url.split("/")[-2],
        "url": url,
    }

    # Load main event metadata
    event_data_path = event / f"{event.stem}.json"
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
    with pastro_path.open("r") as f:
        event_dict["p_bbh"] = json.load(f).get("BBH")

    # Load Aframe latency
    latency_path = event / "latency.log"
    latencies = np.genfromtxt(latency_path, delimiter=",")
    event_dict["latency"] = float(latencies[1, -1])

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
            gpstime = json.load(f)["gpstime"]
        eventdir = outdir / event.stem
        eventdir.mkdir(parents=True, exist_ok=True)
        try:
            get_plots(event, eventdir, gpstime, online_args)
        except FileNotFoundError:
            continue

    return df
