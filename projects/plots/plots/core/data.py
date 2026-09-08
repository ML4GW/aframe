import logging
from dataclasses import dataclass
from pathlib import Path

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet

# Waveforms recovered above this SNR are unphysical, coming from
# IMRPhenomXPHM glitch; see:
# https://git.ligo.org/reed.essick/rpo4-injection-triage/-/blob/main/rpo4a/README.md?ref_type=heads # noqa
MAX_PHYSICAL_SNR = 1e4


def drop_unphysical(
    foreground: RecoveredInjectionSet,
) -> RecoveredInjectionSet:
    """Remove foreground events whose SNR is too large to be physical."""
    mask = foreground.snr > MAX_PHYSICAL_SNR
    num_unphysical = mask.sum()
    if num_unphysical > 0:
        foreground = foreground[~mask]
        logging.info(
            f"Removed {num_unphysical} foreground events "
            f"with SNR > {MAX_PHYSICAL_SNR:,.0f}"
        )
    return foreground


@dataclass
class AnalysisData:
    """The three ledgers an analysis is summarized from."""

    background: EventSet
    foreground: RecoveredInjectionSet
    rejected: InjectionParameterSet

    @classmethod
    def load(
        cls,
        background: Path,
        foreground: Path,
        rejected: Path,
    ) -> "AnalysisData":
        """Read the ledgers and drop unphysical foreground events.

        Args:
            background: HDF5 file readable by `EventSet.read`. Gets sorted by
                detection statistic if it isn't already.
            foreground: HDF5 file readable by `RecoveredInjectionSet.read`
            rejected: HDF5 file readable by `InjectionParameterSet.read`
        """
        logging.info("Reading in inference outputs")
        data = cls(
            background=EventSet.read(background),
            foreground=drop_unphysical(RecoveredInjectionSet.read(foreground)),
            rejected=InjectionParameterSet.read(rejected),
        )

        # Background should be sorted by infer already, but just in case
        if not data.background.is_sorted_by("detection_statistic"):
            background.sort_by("detection_statistic")

        logging.info("Read in:")
        logging.info(f"\t{len(data.background)} background events")
        logging.info(f"\t{len(data.foreground)} foreground events")
        logging.info(f"\t{len(data.rejected)} rejected events")
        return data
