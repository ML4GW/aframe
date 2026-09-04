import logging
from collections.abc import Callable
from pathlib import Path

from utils.cosmology import DEFAULT_COSMOLOGY
from utils.logging import configure_logging

from plots.core.data import AnalysisData
from plots.core.gwtc3 import main as gwtc3_pipeline_sv
from plots.core.sv import (
    SensitiveVolumePlot,
    comparisons_from_gwtc3_curves,
    compute_sensitive_volume,
)
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


def sensitive_volume(
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

    logging.info("Calculating SV vs FAR for GWTC-3 pipelines")
    gwtc3_sv, gwtc3_err = gwtc3_pipeline_sv(
        mass_combos=mass_combos,
        injection_file=injection_file,
        detection_criterion="far",
        detection_thresholds=result.fars,
        output_dir=output_dir,
    )
    comparisons = comparisons_from_gwtc3_curves(
        gwtc3_sv, gwtc3_err, mass_combos
    )
    SensitiveVolumePlot(result, comparisons).save(output_dir)
