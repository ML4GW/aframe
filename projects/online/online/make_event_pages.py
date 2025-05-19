import argparse
import base64
import json
import shutil
import time
from pathlib import Path
from tqdm import tqdm

import h5py
import matplotlib.pyplot as plt
import numpy as np


def html_header(label: str) -> str:
    """
    Generate the HTML header with a title.

    Args:
        label: Title for the HTML page.

    Returns:
        str: HTML header string.
    """
    html_header = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{label}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                margin: 0;
                padding: 20px;
            }}
            .gallery {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }}
            .item {{
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 400px;
                box-sizing: border-box;
            }}
            img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .caption {{
                margin-top: 8px;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h1>{label}</h1>
        <div class="gallery">
    """
    return html_header


def html_footer():
    html_footer = """
        </div>
    </body>
    </html>
    """
    return html_footer


def embed_image(image_path: Path, caption: str) -> str:
    """
    Embed an image in HTML using base64 encoding.

    Args:
        image_path: Path to the image file.
        caption: Caption for the image.

    Returns:
        str: HTML string with the embedded image.
    """
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f'''
        <div class="item">
            <img src="data:image/png;base64,{encoded_string}" alt="{caption}">
            <div class="caption">{caption}</div>
        </div>
    '''


def generate_html(event: Path, outdir: Path):
    """
    Generate an HTML summary page for the event.

    Args:
        eventdir: Directory containing event data.
        outdir: Output directory for the HTML file.
    """
    eventdir = outdir / event.stem
    plotsdir = eventdir / "plots"
    html_file = eventdir / f"{event.stem}.html"

    with open(html_file, "w") as f:
        f.write(html_header(event))
        for png in plotsdir.glob("*.png"):
            caption = png.stem
            f.write(embed_image(png, caption))
        f.write(html_footer())


def process_event_outputs(event: Path, outdir: Path):
    eventdir = outdir / event.stem
    plotsdir = eventdir / "plots"
    eventdir.mkdir(parents=True, exist_ok=True)
    plotsdir.mkdir(parents=True, exist_ok=True)

    with open(event / f"{event.stem}.json", "r") as f:
        gpstime = json.load(f)["gpstime"]

    for png in event.glob("*.png"):
        shutil.copy(png, plotsdir / png.name)

    with h5py.File(event / "output.hdf5", "r") as f:
        time = f["time"][:]
        output = f["output"][:]
        integrate = f["integrated"][:]

    plt.plot(time, output, label="Raw output")
    plt.plot(time, integrate, label="Integrated output")
    plt.axvline(gpstime, color="red", linestyle="--", label="Event time")
    plt.xlabel("GPS time")
    plt.ylabel("Detection statistic")
    plt.legend()
    plt.savefig(plotsdir / "aframe_response.png", dpi=150)
    plt.close()

    asds = np.load(event / "asd.npy")[0]
    freqs = asds[0]
    asds = asds[1:]

    ifos = ["H1", "L1", "V1"]
    for i, ifo in enumerate(ifos[: len(asds)]):
        plt.plot(freqs, asds[i], label=ifo)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("ASD (strain/Hz^0.5)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(plotsdir / "asds.png", dpi=150)
    plt.close()


def main(event_dir: Path, outdir: Path):
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    while True:
        new_events = set(event_dir.iterdir()).difference(set(outdir.iterdir()))
        if new_events:
            for event in tqdm(new_events):
                try:
                    process_event_outputs(event, outdir)
                    generate_html(event, outdir)
                except FileNotFoundError:
                    continue
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process event outputs")
    parser.add_argument(
        "--event_dir",
        type=Path,
        required=True,
        help="Directory containing event directories",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    main(args.event_dir, args.outdir)
