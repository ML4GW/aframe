from pathlib import Path
import pandas as pd
from gwpy.time import tconvert
from datetime import datetime, timezone

from .html import html_header, html_footer, embed_image
from .plotting import event_rate_plots, ifar_plot, latency_plot
from .parse_logs import get_pipeline_status, get_data_status

plot_name_dict = {
    "aframe_latency": "Aframe detection latency",
    "event_rate_past_day": "Event rate over the past day",
    "event_rate_past_week": "Event rate over the past week",
    "event_rate_all_time": "Event rate over all time",
    "ifar_plot": "Zero-lag cumulative distribution vs iFAR",
}


def update_summary_plots(plotsdir: Path, df: pd.DataFrame, tb: float):
    """
    Update summary plots based on the DataFrame of events.

    Args:
        plotsdir: Output directory where the plots will be saved.
        df: DataFrame containing event data.
        tb: Total time in seconds that the pipeline has been running.
    """
    latency_plot(plotsdir, df)
    ifar_plot(plotsdir, df, tb)
    event_rate_plots(plotsdir, df)


def main(
    run_dir: Path, outdir: Path, start_time: float, df: pd.DataFrame, tb: float
):
    html_file = outdir / "summary.html"
    plotsdir = outdir / "plots"
    if not plotsdir.exists():
        plotsdir.mkdir(exist_ok=True, parents=True)

    pipeline_status = get_pipeline_status()
    if pipeline_status:
        pipeline_status = "Online"
        pipeline_color = "green"
    else:
        pipeline_status = "Offline"
        pipeline_color = "red"
    data_status = get_data_status(run_dir)
    if data_status is None:
        data_status = "Aframe offline"
        data_color = "red"
    elif data_status:
        data_status = "Analysis-ready"
        data_color = "green"
    else:
        data_status = "Not analysis-ready"
        data_color = "red"

    update_summary_plots(plotsdir, df, tb)

    date_format = "%Y-%m-%d %H:%M:%S"
    start_time = tconvert(start_time).strftime(date_format)
    current_time = datetime.now(timezone.utc).strftime(date_format)

    with open(html_file, "w") as f:
        f.write(html_header("Aframe Online Status Summary"))
        f.write(f"""
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
            """)

        for name, caption in plot_name_dict.items():
            png = plotsdir / f"{name}.png"
            f.write(embed_image(png, caption))
        f.write(html_footer())
