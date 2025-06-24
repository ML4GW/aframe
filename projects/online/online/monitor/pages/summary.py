from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from gwpy.time import tconvert

from online.monitor.pages import MonitorPage
from online.monitor.utils.plotting import (
    latency_plot,
    ifar_plot,
    event_rate_plots,
)
from online.monitor.utils.parse_logs import data_ready, pipeline_online


class SummaryPage(MonitorPage):
    def __init__(
        self, start_time: float, run_dir: Path, out_dir: Path
    ) -> None:
        super().__init__(run_dir, out_dir)
        self.start_time = start_time
        self.plots_dir = out_dir / "plots"
        if not self.plots_dir.exists():
            self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.html_file = out_dir / "summary.html"

    @property
    def plot_name_dict(self) -> dict:
        return {
            "aframe_latency": "Aframe detection latency",
            "event_rate_past_day": "Event rate over the past day",
            "event_rate_past_week": "Event rate over the past week",
            "event_rate_all_time": "Event rate over all time",
            "ifar_plot": "Zero-lag cumulative distribution vs iFAR",
        }

    def html_body(self):
        pipeline_status, pipeline_color = (
            ("Online", "green") if pipeline_online() else ("Offline", "red")
        )
        data_status = data_ready(self.run_dir)
        data_status, data_color = (
            ("Aframe offline", "red")
            if data_status is None
            else ("Analysis-ready", "green")
            if data_status
            else ("Not analysis-ready", "red")
        )
        date_format = "%Y-%m-%d %H:%M:%S"
        start_time = tconvert(self.start_time).strftime(date_format)
        current_time = datetime.now(timezone.utc).strftime(date_format)
        html_body = f"""
            <style>
                .green {{
                color: green;
                }}
                .red {{
                color: red;
                }}
            </style>
            <body>
                <p> Monitoring events after: {start_time} UTC</p>
                <p> Last updated at: {current_time} UTC</p>
                <p> Aframe:
                    <span class={pipeline_color}>{pipeline_status}</span>
                </p>
                <p> Data: <span class={data_color}>{data_status}</span></p>
            <div class="gallery">
        """
        for name, caption in self.plot_name_dict.items():
            png = self.plots_dir / f"{name}.png"
            html_body += self.embed_image(png, caption)

        return html_body

    def update_summary_plots(self, tb: float):
        """
        Update summary plots based on the DataFrame of events.

        Args:
            tb: Total time in seconds that the pipeline has been running.
        """
        df = pd.read_hdf(self.dataframe_file)
        latency_plot(self.plots_dir, df)
        ifar_plot(self.plots_dir, df, tb)
        event_rate_plots(self.plots_dir, df)

    def write_html(self) -> None:
        """
        Generate the summary HTML page with embedded images and event data.
        """

        with open(self.html_file, "w") as f:
            f.write(self.html_header("Aframe Online Status Summary"))
            f.write(self.html_body())
            f.write(self.html_footer())

    def create(self, tb: float) -> None:
        """
        Create the summary page with the latest plots and event data.

        Args:
            tb: Total time in seconds that the pipeline has been running.
        """
        self.update_summary_plots(tb)
        self.write_html()
