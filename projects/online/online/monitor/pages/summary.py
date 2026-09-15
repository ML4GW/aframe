from datetime import datetime, timezone

import pandas as pd
from gwpy.time import tconvert

from online.monitor.pages import MonitorPage
from online.monitor.utils.plotting import (
    STATE_LABELS,
    duty_cycle_plots,
    latency_plot,
    ifar_plot,
    event_rate_plots,
)
from online.monitor.utils.segments import (
    DETECTOR_FAULT_STATES,
    compute_duty_cycle,
    current_status,
    longest_downtimes,
)
from online.utils.timing import gps_now

SECONDS_PER_DAY = 86400


def format_duration(seconds: float) -> str:
    """Render a number of seconds as e.g. `2d 4h 13m`"""
    seconds = int(seconds)
    days, seconds = divmod(seconds, SECONDS_PER_DAY)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_percent(fraction: float | None) -> str:
    return "--" if fraction is None else f"{100 * fraction:.1f}%"


class SummaryPage(MonitorPage):
    def __init__(
        self,
        start_time: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.start_time = start_time
        if self.start_time is None:
            if self.dataframe_file.exists():
                df = pd.read_hdf(self.dataframe_file)
                self.start_time = min(df["gpstime"])
                self.logger.info(
                    "start_time was None, setting start_time to the oldest "
                    f"event time in {self.dataframe_file}, {self.start_time}"
                )
            else:
                self.start_time = gps_now()
                self.logger.info(
                    "start_time was None and dataframe file does not exist, "
                    "setting start_time to current time"
                )
        self.plots_dir = self.out_dir / "plots"
        if not self.plots_dir.exists():
            self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.html_file = self.out_dir / "summary.html"
        self.segments = None
        self.stats = None

    @property
    def plot_name_dict(self) -> dict:
        return {
            "aframe_latency": "Aframe detection latency",
            "event_rate_past_day": "Event rate over the past day",
            "event_rate_past_week": "Event rate over the past week",
            "event_rate_all_time": "Event rate over all time",
            "ifar_plot": "Zero-lag cumulative distribution vs iFAR",
            "duty_cycle_timeline": "Duty cycle timeline",
            "duty_cycle_trend": "Hourly search duty cycle",
        }

    def duty_cycle_html(self) -> str:
        """
        Duty cycle stats and a table of the downtime we were
        responsible for.
        """
        now = gps_now()
        windows = {
            "Past day": compute_duty_cycle(
                self.segments, start=now - SECONDS_PER_DAY
            ),
            "Past week": compute_duty_cycle(
                self.segments, start=now - 7 * SECONDS_PER_DAY
            ),
            "Since start": self.stats,
        }
        rows = [
            [
                window,
                format_percent(stats["duty_cycle"]),
                format_percent(stats["uptime"]),
                format_duration(stats["livetime"]),
                format_duration(stats["search_downtime"]),
            ]
            for window, stats in windows.items()
        ]
        html = self.html_table(
            [
                "Window",
                "Duty cycle",
                "Uptime",
                "Data analyzed",
                "Unattributed downtime",
            ],
            rows,
            "Search duty cycle",
        )
        html += """
            <p style="max-width: 820px; margin: 0 auto; color: #555;">
            Fraction of analyzable time Aframe was running. Excludes
            detector downtime. Any coincident Aframe/detector downtime
            is counted as Aframe downtime.
            </p>
        """

        downtimes = longest_downtimes(self.segments, n=10)
        if len(downtimes):
            date_format = "%Y-%m-%d %H:%M:%S"
            html += self.html_table(
                ["Start (UTC)", "Duration", "Cause"],
                [
                    [
                        tconvert(row.start).strftime(date_format),
                        format_duration(row.duration),
                        STATE_LABELS.get(row.state, row.state),
                    ]
                    for row in downtimes.itertuples()
                ],
                "Longest stretches of search downtime",
            )
        return html

    def status_html(self) -> str:
        """
        Pipeline and data status read from the search's heartbeat
        """
        state = current_status(self.run_dir)
        if state is None:
            pipeline = ("Offline", "red")
            data = ("Aframe offline", "red")
        elif state in DETECTOR_FAULT_STATES:
            pipeline = ("Online", "green")
            data = ("Not analysis-ready", "red")
        else:
            pipeline = ("Online", "green")
            data = ("Analysis-ready", "green")

        pipeline_status, pipeline_color = pipeline
        data_status, data_color = data
        return f"""
                <p> Aframe:
                    <span class={pipeline_color}>{pipeline_status}</span>
                </p>
                <p> Data: <span class={data_color}>{data_status}</span></p>
        """

    def html_body(self):
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
                {self.status_html()}
                <p> <a href={self.root_url}>Summary root directory</a></p>
                <p> <a href={self.event_root_url}>Event root directory</a></p>
                {self.duty_cycle_html()}
            <div class="gallery">
        """
        for name, caption in self.plot_name_dict.items():
            png = self.plots_dir / f"{name}.png"
            if png.exists():
                html_body += self.embed_image(png, caption)

        return html_body

    def update_summary_plots(self):
        """Update summary plots based on the DataFrame of events."""
        duty_cycle_plots(self.plots_dir, self.segments)

        if not self.dataframe_file.exists():
            self.logger.info("No events detected yet, skipping event plots")
            return

        df = pd.read_hdf(self.dataframe_file)
        df = df[df["gpstime"] >= self.start_time]
        if df.empty:
            self.logger.warning(
                "No events found in the DataFrame after the start time."
            )
            return
        latency_plot(self.plots_dir, df)
        event_rate_plots(self.plots_dir, df)
        ifar_plot(self.plots_dir, df, self.stats["livetime"])

    def write_html(self) -> None:
        self.write_atomic(
            self.html_file,
            self.html_header("Aframe Online Status Summary")
            + self.html_body()
            + self.html_footer(),
        )

    def create(self, segments: pd.DataFrame) -> None:
        """
        Create the summary page with the latest plots and event data.

        Args:
            segments:
                Record of how the search spent its time, from
                `load_segments`.
        """
        self.segments = segments
        self.stats = compute_duty_cycle(segments)
        self.update_summary_plots()
        self.write_html()
