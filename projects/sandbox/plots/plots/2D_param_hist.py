import logging
from pathlib import Path
from typing import Optional

import numpy as np
from bokeh.io import save
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from typeo import scriptify

from aframe.analysis.ledger.events import RecoveredInjectionSet
from aframe.analysis.ledger.injections import InjectionParameterSet
from aframe.logging import configure_logging


@scriptify
def main(
    foreground_file: Path,
    rejected_params: Path,
    output_fname: Path,
    log_file: Optional[Path] = None,
    verbose: bool = False,
):
    configure_logging(log_file, verbose)

    logging.info("Reading in inference outputs")

    foreground = RecoveredInjectionSet.read(foreground_file)
    rejected = InjectionParameterSet.read(rejected_params)
    all_params = {
        k: np.concatenate(((foreground[k], rejected[k])))
        for k in rejected.keys()
    }

    labels = list(rejected.keys())
    labels.remove("phase")

    logging.info("Read in:")
    logging.info(f"\t{len(foreground)} foreground events")
    logging.info(f"\t{len(rejected)} rejected events")
    logging.info(f"\t{len(all_params)} total events")

    fore_plots = []
    rej_plots = []
    all_plots = []
    for x in range(len(labels)):
        for y in range(x, len(labels)):
            # skip 2D plots with equal parameters
            if x == y:
                continue

            fore = create_figure(labels[x], labels[y], foreground, dist="Fore")
            rej = create_figure(
                labels[x], labels[y], rejected_params, dist="Rej"
            )
            all = create_figure(labels[x], labels[y], all_params, dist="All")

            fore_plots.append(fore)
            rej_plots.append(rej)
            all_plots.append(all)

    grid = gridplot(
        [fore_plots, rej_plots, all_plots], toolbar_location="right"
    )
    save(grid, filename=output_fname)


def create_figure(x_var: str, y_var: str, data: dict, dist: str):
    if dist == "Fore":
        title = "2D Normalized Foreground Distribution"
    elif dist == "All":
        title = "2D Normalized Foreground + Rejected Distribution"
    elif dist == "Rej":
        title = "2D Normalized Rejected Distribution"
    else:
        raise ValueError(f"{dist} is not a supported distribution")

    (
        H,
        xe,
        ye,
    ) = np.histogram2d(data[x_var], data[y_var], bins=100, density=True)

    fig = figure(
        title=title,
        x_axis_label=x_var,
        y_axis_label=y_var,
        width=400,
        height=250,
        x_range=(min(xe), max(xe)),
        y_range=(min(ye), max(ye)),
    )
    fig.image(
        image=[np.transpose(H)],
        x=xe[0],
        y=ye[0],
        dw=xe[-1] - xe[0],
        dh=ye[-1] - ye[0],
        palette="Viridis256",
    )
    fig.outline_line_width = 2
    fig.outline_line_color = "black"
    fig.title.text_font_size = "8pt"
    fig.background_fill_color = None

    return fig


if __name__ == "__main__":
    main()
