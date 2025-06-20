from pathlib import Path

from .html import html_header, html_footer, embed_image

IFOS = ["H1", "L1", "V1"]

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
plot_name_dict |= {f"{ifo}_qtransform": f"{ifo} Q-transform" for ifo in IFOS}


def main(event: Path, url: str, outdir: Path):
    """
    Generate an HTML summary page for the event.

    Args:
        eventdir: Directory containing event data.
        url: URL to GraceDB page of event
        outdir: Output directory for the HTML file.
    """
    eventdir = outdir / event.stem
    plotsdir = eventdir / "plots"
    html_file = eventdir / f"{event.stem}.html"

    with open(html_file, "w") as f:
        f.write(html_header(event.name))

        f.write(
            f"""
            <body>
                <h1>
                    <a href={url}>{url}</a>
                </h1>
                <div class="gallery">
            """
        )

        for png in sorted(plotsdir.glob("*.png")):
            caption = plot_name_dict[png.stem]
            f.write(embed_image(png, caption))
        f.write(html_footer())
