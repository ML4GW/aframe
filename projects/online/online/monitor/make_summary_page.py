from pathlib import Path
import psutil
import pandas as pd
from gwpy.time import tconvert
from datetime import datetime, timezone

from .html import html_header, html_footer, embed_image
from .plotting import latency_plot

plot_name_dict = {
    "aframe_latency": "Aframe detection latency",
}


def get_pipeline_status(expected_process_count: int = 6):
    online_processes = 0
    for p in psutil.process_iter(["username", "name"]):
        if p.info["username"] == "aframe" and p.info["name"] == "online":
            online_processes += 1
    return online_processes == expected_process_count


def get_data_status(run_dir: Path):
    if not get_pipeline_status():
        return
    log_dir = sorted((run_dir / "output" / "logs").iterdir())[-1]
    log_file = sorted(log_dir.iterdir())[-1]

    # Given the current logging, I don't think
    # there's anything more efficient than loading
    # the entire log file. The files change each
    # day, so this shouldn't ever be too bad
    with open(log_file, "r") as f:
        lines = f.readlines()

    failure_lines = [
        "H1 exiting analysis ready mode\n",
        "L1 exiting analysis ready mode\n",
        "H1 not analysis ready\n",
        "L1 not analysis ready\n",
    ]

    ready_line = "is ready again, resetting states\n"

    # Go through the lines in reverse order and
    # return the state based on whichever condition
    # we find first
    for line in lines[::-1]:
        if any(line.endswith(failure) for failure in failure_lines):
            return False
        if line.endswith(ready_line):
            return True
    return True


def update_summary_plots(plotsdir: Path, df: pd.DataFrame):
    """
    Update summary plots based on the DataFrame of events.

    Args:
        plotsdir: Output directory where the plots will be saved.
        df: DataFrame containing event data.
    """
    latency_plot(plotsdir, df)


def main(
    run_dir: Path, outdir: Path, start_time: float, df: pd.DataFrame = None
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

    if df is not None:
        update_summary_plots(plotsdir, df)

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
            </body>
            """)
        for png in sorted(plotsdir.glob("*.png")):
            caption = plot_name_dict[png.stem]
            f.write(embed_image(png, caption))
        f.write(html_footer())
