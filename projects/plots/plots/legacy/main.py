import logging
from collections.abc import Callable
from pathlib import Path

import jsonargparse
from bokeh.io import save
from bokeh.layouts import gridplot
from utils.cosmology import DEFAULT_COSMOLOGY
from utils.logging import configure_logging

from plots.core import style
from plots.core.data import AnalysisData
from plots.core.gwtc3 import main as gwtc3_pipeline_sv
from plots.core.sv import compute_sensitive_volume
from plots.vetos import (
    GATE_PATHS,
    VETO_CATEGORIES,
    VETO_DEFINER_FILE,
    VetoParser,
    get_catalog_vetos,
)

logging.getLogger("urllib3").setLevel(logging.WARNING)


def _apply_vetos(background, foreground, vetos, ifos, start, stop):
    """Filter background and foreground events through veto categories."""
    veto_parser = VetoParser(VETO_DEFINER_FILE, GATE_PATHS, start, stop, ifos)
    catalog_vetos = get_catalog_vetos(start, stop)
    for cat in vetos:
        for i, ifo in enumerate(ifos):
            if cat == "CATALOG":
                cat_vetos = catalog_vetos
            else:
                cat_vetos = veto_parser.get_vetos(cat)[ifo]
            back_count = len(background)
            fore_count = len(foreground)
            if len(cat_vetos) > 0:
                background = background.apply_vetos(cat_vetos, i)
                foreground = foreground.apply_vetos(cat_vetos, i)
            logging.info(
                f"\t{back_count - len(background)} {cat} "
                f"background events removed for ifo {ifo}"
            )
            logging.info(
                f"\t{fore_count - len(foreground)} {cat} "
                f"foreground events removed for ifo {ifo}"
            )
    return background, foreground


def main(
    background: Path,
    foreground: Path,
    rejected_params: Path,
    ifos: list[str],
    mass_combos: list[tuple],
    source_prior: Callable,
    output_dir: Path,
    log_file: Path | None = None,
    dt: float | None = None,
    max_far: float = 365,
    sigma: float = 0.1,
    verbose: bool = False,
    vetos: list[VETO_CATEGORIES] | None = None,
    injection_file: Path | None = None,
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
            The maximum FAR to compute the sensitive volume out to in
            units of years^-1
        sigma:
            The width of the log normal mass distribution to use
        verbose:
            If true, log at the debug level
        injection_file:
            Path to the LVK O3 sensitivity injection set used for the
            GWTC-3 comparison curves. If not provided, it is downloaded
            from Zenodo and cached under `~/.aframe/cache`.
    """
    configure_logging(log_file, verbose)
    data = AnalysisData.load(background, foreground, rejected_params)
    background = data.background
    foreground = data.foreground

    if len(background):
        start, stop = (
            background.detection_time.min(),
            background.detection_time.max(),
        )
    else:
        start = stop = 0.0
    logging.info(f"Loading in vetoes from {start} to {stop}")

    # optionally apply vetos if the user passed a list of veto categories
    if vetos is not None:
        background, foreground = _apply_vetos(
            background, foreground, vetos, ifos, start, stop
        )
        data = AnalysisData(background, foreground, data.rejected)

    source, _ = source_prior(DEFAULT_COSMOLOGY)
    result = compute_sensitive_volume(
        data,
        mass_combos=mass_combos,
        source_prior=source,
        dt=dt,
        max_far=max_far,
        sigma=sigma,
    )
    result.write(output_dir / "sensitive_volume.hdf5")
    aframe_sv, aframe_err = result.sv, result.err
    fars = result.fars

    logging.info("Calculating SV vs FAR for GWTC-3 pipelines")
    gwtc3_sv, gwtc3_err = gwtc3_pipeline_sv(
        mass_combos=mass_combos,
        injection_file=injection_file,
        detection_criterion="far",
        detection_thresholds=fars,
        output_dir=output_dir,
    )

    plots = style.make_grid(mass_combos)
    for i, p in enumerate(plots):
        color = style.palette[0]
        # only include a legend on the top left
        kwargs = {}
        if i == 0:
            kwargs["legend_label"] = "aframe"
        p.line(fars, aframe_sv[i], line_width=1.5, line_color=color, **kwargs)
        style.plot_err_bands(
            p,
            fars,
            aframe_sv[i],
            aframe_err[i],
            line_color=color,
            line_width=0.8,
            fill_color=color,
            fill_alpha=0.4,
        )

        for pipeline, color in zip(
            gwtc3_sv.keys(), style.palette[1:], strict=False
        ):
            m1, m2 = mass_combos[i]
            mass_key = f"{m1}-{m2}"
            sv = gwtc3_sv[pipeline][mass_key]
            err = gwtc3_err[pipeline][mass_key]

            if i == 0:
                kwargs["legend_label"] = pipeline
            p.line(fars, sv, line_width=1.5, line_color=color, **kwargs)
            style.plot_err_bands(
                p,
                fars,
                sv,
                err,
                line_color=color,
                line_width=0.8,
                fill_color=color,
                fill_alpha=0.4,
            )

    # style the legend on the top left plot
    legend = plots[0].legend
    legend.ncols = 2
    # style legend position
    legend.location = "top_left"
    legend.margin = 4
    legend.padding = 2

    # style individual glyphs
    legend.glyph_height = 6
    legend.label_text_font_size = "8pt"
    legend.label_height = 8

    grid = gridplot(plots, toolbar_location="right", ncols=2)
    save(grid, filename=output_dir / "sensitive_volume.html")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    args = parser.parse_args()
    main(**vars(args))
