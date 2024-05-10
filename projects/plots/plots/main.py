import logging
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import jsonargparse
import numpy as np
from astropy.cosmology import Planck15 as cosmology
from bokeh.io import save
from bokeh.layouts import gridplot
from plots import compute, tools
from plots.gwtc3 import catalog_results
from plots.vetoes import VetoParser, get_catalog_vetoes

from aframe.ledger.events import EventSet, RecoveredInjectionSet
from aframe.ledger.injections import InjectionParameterSet
from aframe.priors.priors import log_normal_masses
from utils.logging import configure_logging

logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_prob(prior, ledger):
    sample = dict(mass_1=ledger.mass_1, mass_2=ledger.mass_2)
    return prior.prob(sample, axis=0)


def normalize_path(path):
    path = Path(path)
    if not path.is_absolute():
        return Path(__file__).resolve().parent / path
    return path


VETO_DEFINER_FILE = normalize_path("./vetoes/H1L1-HOFT_C01_O3_CBC.xml")
GATE_PATHS = {
    "H1": normalize_path("./vetoes/H1-O3_GATES_1238166018-31197600.txt"),
    "L1": normalize_path("./vetoes/L1-O3_GATES_1238166018-31197600.txt"),
}


def main(
    background: Path,
    foreground: Path,
    rejected_params: Path,
    mass_combos: List[tuple],
    source_prior: Callable,
    output_dir: Path,
    log_file: Optional[Path] = None,
    dt: Optional[float] = None,
    max_far: float = 1000,
    sigma: float = 0.1,
    verbose: bool = False,
):
    """
    Compute and plot the sensitive volume of an aframe analysis

    Args:
        background:
            Path to the background event set. Should be an HDF5 file
            readable by `ledger.events.EventSet.read`
        foreground:
            Path to the foreground event set. Should be an HDF5 file
            readable by `ledger.injections.RecoveredInjectionSet.read`
        rejected_params:
            Path to the rejected parameter set. Should be an HDF5 file
            readable by `ledger.injections.InjectionParameterSet.read`
        output_dir:
            Path to the directory to save the output plots and data
        log_file:
            Path to the log file. If not provided, will log to stdout
        dt:
            If provided, enforce a recovery time delta of `dt` seconds
            between injected and recovered events. Note that your `dt`
            should be greater than 1 / `inference_sampling_rate`.
        max_far:
            The maximum FAR to compute the sensitive volume out to
        sigma:
            The width of the log normal mass distribution to use
        verbose:
            If true, log at the debug level
    """

    configure_logging(log_file, verbose)
    logging.info("Reading in inference outputs")
    background = EventSet.read(background)
    foreground = RecoveredInjectionSet.read(foreground)
    rejected_params = InjectionParameterSet.read(rejected_params)

    for i in range(2):
        mass = f"mass_{i + 1}"
        for ledger in [foreground, rejected_params]:
            val = getattr(ledger, mass)
            setattr(ledger, mass, val / (1 + ledger.redshift))
    logging.info("Read in:")
    logging.info(f"\t{len(background)} background events")
    logging.info(f"\t{len(foreground)} foreground events")
    logging.info(f"\t{len(rejected_params)} rejected events")

    start, stop = (
        background.detection_time.min(),
        background.detection_time.max(),
    )
    logging.info(f"Loading in vetoes from {start} to {stop}")

    ifos = ["H1", "L1"]

    veto_parser = VetoParser(
        VETO_DEFINER_FILE,
        GATE_PATHS,
        start,
        stop,
        ifos,
    )

    catalog_vetos = get_catalog_vetoes(start, stop)
    categories = ["CAT1", "CAT2", "CAT3", "GATES", "CATALOG"]
    for cat in categories:
        for i, ifo in enumerate(ifos):
            if cat == "CATALOG":
                vetos = catalog_vetos
            else:
                vetos = veto_parser.get_vetoes(cat)[ifo]
            back_count = len(background)
            fore_count = len(foreground)
            if len(vetos) > 0:
                background = background.apply_vetos(vetos, i)
                foreground = foreground.apply_vetos(vetos, i)
            logging.info(
                f"\t{back_count - len(background)} {cat} "
                f"background events removed for ifo {ifo}"
            )
            logging.info(
                f"\t{fore_count - len(foreground)} {cat} "
                f"foreground events removed for ifo {ifo}"
            )

    logging.info("Computing data likelihood under source prior")
    source, _ = source_prior(cosmology)
    source_probs = get_prob(source, foreground)
    source_rejected_probs = get_prob(source, rejected_params)

    logging.info("Computing maximum astrophysical volume")
    zprior = source["redshift"]
    zmin, zmax = zprior.minimum, zprior.maximum

    try:
        decprior = source["dec"]
    except KeyError:
        decrange = None
    else:
        decrange = (decprior.minimum, decprior.maximum)
    v0 = tools.get_astrophysical_volume(zmin, zmax, cosmology, decrange)
    v0 /= 10**9

    Tb = background.Tb / tools.SECONDS_PER_MONTH
    max_events = int(max_far * Tb)
    x = np.arange(1, max_events + 1) / Tb
    thresholds = np.sort(background.detection_statistic)[-max_events:][::-1]

    weights = np.zeros((len(mass_combos), len(source_probs)))
    for i, combo in enumerate(mass_combos):
        logging.info(f"Computing likelihoods under {combo} log normal")
        prior, _ = log_normal_masses(*combo, sigma=sigma, cosmology=cosmology)
        prob = get_prob(prior, foreground)
        rejected_prob = get_prob(prior, rejected_params)

        weight = prob / source_probs

        rejected_weights = rejected_prob / source_rejected_probs
        norm = weight.sum() + rejected_weights.sum()
        weight /= norm

        # finally, enforce recovery time delta by setting weights to 0
        # for events outside of the delta t
        if dt is not None:
            logging.info(f"Enforcing recovery time delta of {dt} seconds")
            mask = (
                np.abs(foreground.detection_time - foreground.injection_time)
                <= dt
            )
            weight[~mask] = 0

        weights[i] = weight

    logging.info("Computing sensitive volume at thresholds")
    y, err = compute.sensitive_volume(
        foreground.detection_statistic, weights, thresholds
    )
    y *= v0
    err *= v0

    output_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(output_dir / "sensitive_volume.h5", "w") as f:
        f.create_dataset("thresholds", data=thresholds)
        f.create_dataset("fars", data=x)
        for i, combo in enumerate(mass_combos):
            g = f.create_group("-".join(map(str, combo)))
            g.create_dataset("sv", data=y[i])
            g.create_dataset("err", data=err[i])

    plots = tools.make_grid(mass_combos)
    for i, (p, color) in enumerate(zip(plots, tools.palette)):
        p.line(x, y[i], line_width=1.5, line_color=color)
        tools.plot_err_bands(
            p,
            x,
            y[i],
            err[i],
            line_color=color,
            line_width=0.8,
            fill_color=color,
            fill_alpha=0.4,
        )

        for pipeline, data in catalog_results.items():
            # convert VT to volume by dividing out years
            vt = data["vt"][mass_combos[i]]
            v = vt * 365 / data["Tb"]

            # only include a legend on the top left
            kwargs = {}
            if i == 0:
                kwargs["legend_label"] = pipeline
            p.line(
                [x[0], x[-1]],
                [v, v],
                line_color="#333333",
                line_dash=data["dash"],
                line_alpha=0.7,
                line_width=2,
                **kwargs,
            )

            # style the legend on the top left plot
            if i == 0:
                # style legend position
                p.legend.location = "top_left"
                p.legend.margin = 4
                p.legend.padding = 2

                # style individual glyphs
                p.legend.glyph_height = 6
                p.legend.label_text_font_size = "8pt"
                p.legend.label_height = 8

                # style title
                p.legend.title = "GWTC-3 comparisons"
                p.legend.title_text_font_size = "9pt"
                p.legend.title_text_font_style = "bold"

    grid = gridplot(plots, toolbar_location="right", ncols=2)
    save(grid, filename=output_dir / "sensitive_volume.html")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    args = parser.parse_args()
    main(**vars(args))
