from datetime import datetime, timezone

from gwpy.time import to_gps


def gps_now() -> float:
    """The current GPS time in seconds."""
    return float(to_gps(datetime.now(timezone.utc)))
