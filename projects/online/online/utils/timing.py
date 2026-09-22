from datetime import UTC, datetime

from gwpy.time import to_gps


def gps_now() -> float:
    """The current GPS time in seconds."""
    return float(to_gps(datetime.now(UTC)))
