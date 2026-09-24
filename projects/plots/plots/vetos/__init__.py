from dataclasses import dataclass
from pathlib import Path

from .vetos import (
    DEFAULT_SEGMENT_SERVER,
    VETO_CATEGORIES,
    VetoParser,
    gates_to_veto_segments,
    get_catalog_vetos,
)

_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Epoch:
    """A veto definer and its gate files, with the times they apply to."""

    start: float
    stop: float
    veto_definer_file: Path
    gate_paths: dict[str, Path]

    def covers(self, start: float, stop: float) -> bool:
        return self.start <= start and stop <= self.stop


# Sourced from https://git.ligo.org/detchar/veto-definitions (cbc/{O3,O4}).
# O4 is CAT1 only, and has no gate files.
EPOCHS = {
    "O3": Epoch(
        start=1235750418,
        stop=1269363618,
        veto_definer_file=_PACKAGE_DIR / "H1L1-HOFT_C01_O3_CBC.xml",
        gate_paths={
            "H1": _PACKAGE_DIR / "H1-O3_GATES_1238166018-31197600.txt",
            "L1": _PACKAGE_DIR / "L1-O3_GATES_1238166018-31197600.txt",
        },
    ),
    "O4": Epoch(
        start=1366556418,
        stop=1447516818,
        veto_definer_file=_PACKAGE_DIR / "H1L1V1-HOFT_C01_O4_CBC.xml",
        gate_paths={},
    ),
}


def get_epoch(start: float, stop: float) -> Epoch:
    """The epoch whose veto definitions apply to `[start, stop)`.

    Raises:
        ValueError: if no shipped epoch covers the span, which includes a
            span straddling two observing runs.
    """
    matches = [e for e in EPOCHS.values() if e.covers(start, stop)]
    if len(matches) != 1:
        known = ", ".join(
            f"{name} [{e.start:.0f}, {e.stop:.0f}]"
            for name, e in EPOCHS.items()
        )
        raise ValueError(
            f"No veto definitions cover [{start:.0f}, {stop:.0f}); "
            f"shipped epochs are {known}. Pass veto_definer_file and "
            "gate_paths explicitly to use definitions from elsewhere."
        )
    return matches[0]


# Defaults kept for callers that predate `get_epoch`.
VETO_DEFINER_FILE = EPOCHS["O3"].veto_definer_file
GATE_PATHS = EPOCHS["O3"].gate_paths

__all__ = [
    "DEFAULT_SEGMENT_SERVER",
    "EPOCHS",
    "GATE_PATHS",
    "VETO_CATEGORIES",
    "VETO_DEFINER_FILE",
    "Epoch",
    "VetoParser",
    "gates_to_veto_segments",
    "get_catalog_vetos",
    "get_epoch",
]
