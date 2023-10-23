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

    hist_dict_fore = {
        k: np.histogram(foreground[k], bins=100, density=True) for k in labels
    }
    hist_dict_rej = {
        k: np.histogram(rejected[k], bins=100, density=True) for k in labels
    }
    hist_dict_all = {
        k: np.histogram(all_params[k], bins=100, density=True) for k in labels
    }

    fore_plots = []
    rej_plots = []
    all_plots = []
    for lab in labels:
        fore = create_figure(lab, data=hist_dict_fore, dist="Fore")
        rej = create_figure(lab, data=hist_dict_rej, dist="Rej")
        all = create_figure(lab, data=hist_dict_all, dist="All")

        fore_plots.append(fore)
        rej_plots.append(rej)
        all_plots.append(all)

    grid = gridplot(
        [fore_plots, rej_plots, all_plots], toolbar_location="right"
    )
    save(grid, filename=output_fname)


def create_figure(x_var: str, data: dict, dist: str):
    if dist == "Fore":
        title = "Normalized Foreground Distribution of {}".format(x_var)
    elif dist == "All":
        title = "Normalized Foreground + Rejected Distribution of {}".format(
            x_var
        )
    elif dist == "Rej":
        title = "Normalized Rejected Distribution of {}".format(x_var)
    else:
        raise ValueError(f"{dist} is not a supported distribution")

    hist_dict = data
    hist = hist_dict[x_var][0]
    edges = hist_dict[x_var][1]
    fig = figure(title=title, x_axis_label=x_var, width=350, height=250)
    fig.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="lightskyblue",
        line_color="white",
    )

    fig.y_range.start = 0
    fig.outline_line_width = 2
    fig.outline_line_color = "black"
    fig.title.text_font_size = "8pt"

    return fig


if __name__ == "__main__":
    main()
