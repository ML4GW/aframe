from pathlib import Path

from .vetos import (
    VETO_CATEGORIES,
    VetoParser,
    gates_to_veto_segments,
    get_catalog_vetos,
)

_PACKAGE_DIR = Path(__file__).resolve().parent

VETO_DEFINER_FILE = _PACKAGE_DIR / "H1L1-HOFT_C01_O3_CBC.xml"
GATE_PATHS = {
    "H1": _PACKAGE_DIR / "H1-O3_GATES_1238166018-31197600.txt",
    "L1": _PACKAGE_DIR / "L1-O3_GATES_1238166018-31197600.txt",
}
