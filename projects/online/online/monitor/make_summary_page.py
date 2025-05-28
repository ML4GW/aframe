from pathlib import Path
import psutil


def get_pipeline_status(expected_process_count: int = 5):
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
        "H1 exiting analysis ready mode",
        "L1 exiting analysis ready mode",
        "H1 not analysis ready",
        "L1 not analysis ready",
    ]

    for line in lines:
        if any(line.endswith(failure) for failure in failure_lines):
            return False
    return True


def generate_html(run_dir: Path, outdir: Path):
    html_file = outdir / "summary.html"

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

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Status Page</title>
    <style>
        .green {{
        color: green;
        }}
        .red {{
        color: red;
        }}
    </style>
    </head>
    <body>
    <p>Aframe: <span class={pipeline_color}>{pipeline_status}</span></p>
    <p>Data: <span class={data_color}>{data_status}</span></p>
    </body>
    </html>
    """

    with open(html_file, "w") as f:
        f.write(html)


def main(run_dir: Path, outdir: Path):
    generate_html(run_dir, outdir)
