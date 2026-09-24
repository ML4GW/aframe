import logging
from collections.abc import Callable
from pathlib import Path

from utils.cosmology import DEFAULT_COSMOLOGY
from utils.logging import configure_logging

from plots.core.constants import DEFAULT_NUM_FAR_POINTS
from plots.core.data import AnalysisData
from plots.core.gwtc3 import main as gwtc3_pipeline_sv
from plots.core.sv import (
    SensitiveVolumePlot,
    comparisons_from_gwtc3_curves,
    compute_sensitive_volume,
)
from plots.vetos import VETO_CATEGORIES
from plots.vetos.masks import (
    combine_masks,
    compute_veto_masks,
    load_or_fetch_segments,
    read_segments,
)

logging.getLogger("urllib3").setLevel(logging.WARNING)


def analysis_span(background) -> tuple[float, float]:
    """GPS span of the data behind `background`, including timeslides."""
    if not len(background):
        return 0.0, 0.0
    shifts = background.shift
    return (
        background.detection_time.min() + min(shifts.min(), 0),
        background.detection_time.max() + max(shifts.max(), 0),
    )


def _apply_vetos(
    background,
    foreground,
    vetos,
    ifos,
    start,
    stop,
    veto_definer_file=None,
    veto_segments=None,
):
    """Filter background and foreground events through veto categories.

    The background's livetime is reduced by the fraction of injections the
    same vetoes remove. Vetoing a time in one IFO removes a different set of
    coincidences in every timeslide, so that fraction, and not the fraction
    of background events removed, estimates the livetime lost.
    """
    if veto_segments is None:
        segments = load_or_fetch_segments(
            vetos, ifos, start, stop, veto_definer_file=veto_definer_file
        )
    else:
        logging.info(f"Reading veto segments from {veto_segments}")
        segments = read_segments(veto_segments)
        missing = set(vetos) - set(segments)
        if missing:
            raise ValueError(
                f"{veto_segments} has no segments for {sorted(missing)}"
            )

    logging.info("Computing background veto masks")
    back_masks = compute_veto_masks(background, vetos, ifos, segments)
    logging.info("Computing foreground veto masks")
    fore_masks = compute_veto_masks(foreground, vetos, ifos, segments)

    fore_vetoed = combine_masks(fore_masks, vetos)
    deadtime = fore_vetoed.mean() if len(fore_vetoed) else 0.0

    background = background[~combine_masks(back_masks, vetos)]
    foreground = foreground[~fore_vetoed]

    logging.info(
        f"Vetoes removed {100 * deadtime:.2f}% of injections; "
        f"scaling Tb from {background.Tb:.0f}s to "
        f"{background.Tb * (1 - deadtime):.0f}s"
    )
    background.Tb *= 1 - deadtime
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
    num_far_points: int = DEFAULT_NUM_FAR_POINTS,
    sigma: float = 0.1,
    verbose: bool = False,
    vetos: list[VETO_CATEGORIES] | None = None,
    veto_definer_file: Path | None = None,
    veto_segments: Path | None = None,
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
        num_far_points:
            Number of points in the FAR grid to compute the sensitive
            volume at
        sigma:
            The width of the log normal mass distribution to use
        verbose:
            If true, log at the debug level
        vetos:
            Veto categories to apply before computing the sensitive volume.
            Note that not every observing run defines every category: the
            O4 CBC definitions are CAT1 only.
        veto_definer_file:
            Path to a LIGO_LW veto definer to use instead of the definitions
            shipped for the observing run the data falls in.
        veto_segments:
            Path to segments written by
            `plots.vetos.masks.load_or_fetch_segments`, used as-is instead
            of querying the segment database. If not provided, segments
            are queried and cached under `~/.aframe/cache`.
        injection_file:
            Path to the LVK O3 sensitivity injection set used for the
            GWTC-3 comparison curves. If not provided, it is downloaded
            from Zenodo and cached under `~/.aframe/cache`.
    """
    configure_logging(log_file, verbose)
    data = AnalysisData.load(background, foreground, rejected_params)
    background = data.background
    foreground = data.foreground

    start, stop = analysis_span(background)
    logging.info(f"Loading in vetoes from {start} to {stop}")

    # optionally apply vetos if the user passed a list of veto categories
    if vetos is not None:
        background, foreground = _apply_vetos(
            background,
            foreground,
            vetos,
            ifos,
            start,
            stop,
            veto_definer_file=veto_definer_file,
            veto_segments=veto_segments,
        )
        data = AnalysisData(background, foreground, data.rejected)

    source, _ = source_prior(DEFAULT_COSMOLOGY)
    result = compute_sensitive_volume(
        data,
        mass_combos=mass_combos,
        source_prior=source,
        dt=dt,
        max_far=max_far,
        num_far_points=num_far_points,
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
