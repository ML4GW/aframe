import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from gwpy.time import tconvert

from online.monitor.pages import MonitorPage
from online.monitor.utils.plotting import (
    aframe_response_plot,
    asd_plot,
    q_plots,
)

IFOS = ["H1", "L1", "V1"]


class EventPage(MonitorPage):
    def __init__(
        self, source_event: Path, online_args: dict, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_event = source_event
        self.online_args = online_args
        self.event_dir = self.output_event_dir / source_event.name
        self.plots_dir = self.event_dir / "plots"
        if not self.plots_dir.exists():
            self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.html_file = self.event_dir / "event.html"
        self.event_data = self.get_event_data(source_event)

    @property
    def plot_name_dict(self) -> dict:
        # Some repeated captions to account for different
        # file names we've used in the past. Eventually,
        # we can remove the older ones
        plot_name_dict = {
            "aframe_response": "Aframe response",
            "amplfi.flattened": "AMPLFI low-latency skymap",
            "amplfi.histogram": "AMPLFI low-latency skymap",
            "amplfi.mollweide": "AMPLFI low-latency skymap",
            "amplfi.kde": "AMPLFI ligo-skymap-from-samples",
            "amplfi.multiorder": "AMPLFI ligo-skymap-from-samples",
            "ligo.skymap.mollweide": "AMPLFI ligo-skymap-from-samples",
            "asds": "Background ASDs",
            "corner_plot": "Source parameter posteriors",
        }
        plot_name_dict |= {
            f"{ifo}_qtransform": f"{ifo} Q-transform" for ifo in IFOS
        }
        return plot_name_dict

    def html_body(self):
        url = self.event_data["url"]
        html_body = f"""
            <body>
                <h1>
                    <a href={url}>{url}</a>
                </h1>
                <div class="gallery">
            """

        for png in sorted(self.plots_dir.glob("*.png")):
            caption = self.plot_name_dict[png.stem]
            html_body += self.embed_image(png, caption)

        return html_body

    def missing_plots(self):
        source_plots = {png.name for png in self.source_event.glob("*.png")}
        existing_plots = {png.name for png in self.plots_dir.glob("*.png")}
        missing_plots = source_plots - existing_plots
        if missing_plots:
            return True
        return False

    def get_plots(self):
        # Copy existing PNG files to the plots directory
        for png in self.source_event.glob("*.png"):
            shutil.copy(png, self.plots_dir / png.name)

        # Generate specific plots
        aframe_response_plot(
            self.source_event,
            self.plots_dir,
            self.event_data["gpstime"],
            self.event_data["far"],
        )
        asd_plot(self.source_event, self.plots_dir)
        q_plots(
            self.source_event,
            self.plots_dir,
            self.event_data["gpstime"],
            self.online_args,
        )

    def get_event_data(self, event: Path) -> pd.DataFrame:
        event_dict = {
            "event": None,
            "url": None,
            "gpstime": None,
            "datetime": None,
            "far": None,
            "p_bbh": None,
            "total latency": None,
            "frame write latency": None,
            "aframe latency": None,
            "chirp_mass_mean": None,
            "chirp_mass_median": None,
            "mass_ratio_mean": None,
            "mass_ratio_median": None,
            "distance_mean": None,
            "distance_median": None,
        }

        # Build event_dict from files
        url_path = event / "gracedb_url.txt"
        if url_path.exists():
            with url_path.open("r") as f:
                url = f.readline().strip()  # Strip newline
            event_dict["event"] = url.split("/")[-2]
            event_dict["url"] = url

        # Load main event metadata
        event_data_path = event / f"{event.stem}.json"
        if event_data_path.exists():
            with event_data_path.open("r") as f:
                event_data = json.load(f)
            event_dict["gpstime"] = event_data.get("gpstime")
            event_dict["datetime"] = tconvert(event_data.get("gpstime"))
            event_dict["far"] = event_data.get("far")

        # Load p_astro
        pastro_path = event / "aframe.p_astro.json"
        if pastro_path.exists():
            with pastro_path.open("r") as f:
                event_dict["p_bbh"] = json.load(f).get("BBH")

        # Load Aframe latency
        latency_path = event / "latency.log"
        if latency_path.exists():
            latencies = np.genfromtxt(latency_path, delimiter=",")[1]
            event_dict["total latency"] = latencies[0]
            event_dict["frame write latency"] = latencies[1]
            event_dict["aframe latency"] = latencies[2]

        # Load posterior data
        posterior_path = event / "amplfi.posterior_samples.hdf5"
        if posterior_path.exists():
            with h5py.File(posterior_path, "r") as f:
                posterior = f["posterior_samples"][:]
                chirp_mass = posterior["chirp_mass"]
                mass_ratio = posterior["mass_ratio"]
                distance = posterior["luminosity_distance"]
                event_dict["chirp_mass_mean"] = np.mean(chirp_mass)
                event_dict["chirp_mass_median"] = np.median(chirp_mass)
                event_dict["mass_ratio_mean"] = np.mean(mass_ratio)
                event_dict["mass_ratio_median"] = np.median(mass_ratio)
                event_dict["distance_mean"] = np.mean(distance)
                event_dict["distance_median"] = np.median(distance)

        return event_dict

    def update_dataframe(self) -> None:
        # Append to the existing DataFrame or create a new one
        if self.dataframe_file.exists():
            prev_df = pd.read_hdf(self.dataframe_file)
            if self.event_data["event"] in prev_df["event"].values:
                # If the event already exists, update it
                idx = np.argwhere(
                    prev_df["event"] == self.event_data["event"]
                )[0, 0]
                prev_df.loc[idx] = self.event_data
                df = prev_df
            else:
                # If the event is new, append it
                df = pd.concat(
                    [prev_df, pd.DataFrame([self.event_data])],
                    ignore_index=True,
                )
        else:
            df = pd.DataFrame([self.event_data])

        df.to_hdf(self.dataframe_file, key="event_data", index=False)

    def write_html(self) -> None:
        with open(self.html_file, "w") as f:
            f.write(self.html_header(self.source_event.name))
            f.write(self.html_body())
            f.write(self.html_footer())

    def create(self) -> None:
        """
        Create or update the event page
        """
        self.event_dir.mkdir(exist_ok=True, parents=True)

        if self.missing_plots():
            self.logger.info(f"Processing {self.source_event.name}")
            self.get_plots()
            self.write_html()
            self.update_dataframe()
