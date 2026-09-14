import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from online.utils.segments import (
    COLUMNS,
    CURRENT_FILE,
    SEGMENTS_FILE,
    PipelineState,
)
from online.utils.timing import gps_now

DETECTOR_FAULT_STATES = (PipelineState.NOT_READY, PipelineState.MISSING_DATA)
SEARCH_FAULT_STATES = (
    PipelineState.WARMUP,
    PipelineState.STARTUP,
    PipelineState.SEARCH_DOWN,
)

# How long the heartbeat can go unwritten before calling the search dead.
STALE_SECONDS = 30.0

logger = logging.getLogger("monitor-segments")


def segment_dir(run_dir: Path) -> Path:
    return run_dir / "output" / "segments"


def read_heartbeat(run_dir: Path) -> Optional[dict]:
    """
    Read the search's heartbeat file, or None if it isn't there. The
    search rewrites it in place, so a read can land on a half-written
    one; that clears up by the next monitor cycle.
    """
    current_file = segment_dir(run_dir) / CURRENT_FILE
    if not current_file.exists():
        return None
    try:
        with open(current_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not read heartbeat file {current_file}")
        return None


def current_status(run_dir: Path) -> dict:
    """
    The current state of the search. None if the search is stale
    or not running.
    """
    heartbeat = read_heartbeat(run_dir)
    running = (
        heartbeat is not None
        and gps_now() - heartbeat["heartbeat"] <= STALE_SECONDS
    )
    return {
        "running": running,
        "state": heartbeat["state"] if running else None,
    }


def load_segments(
    run_dir: Path, start_time: Optional[float] = None
) -> pd.DataFrame:
    """
    Load the search's segment record, filling in the stretches it
    couldn't write itself.

    Args:
        run_dir: Root directory of the online search
        start_time:
            The earliest GPS time to consider. A segment straddling
            this time is cropped.
    """
    segment_file = segment_dir(run_dir) / SEGMENTS_FILE
    if not segment_file.exists():
        raise FileNotFoundError(f"No segment record at {segment_file}.")

    df = pd.read_csv(segment_file, dtype={"ifos_ready": str})
    df["ifos_ready"] = df["ifos_ready"].fillna("")

    heartbeat = read_heartbeat(run_dir)
    if heartbeat is not None:
        open_segment = pd.DataFrame([{c: heartbeat[c] for c in COLUMNS}])
        df = pd.concat([df, open_segment], ignore_index=True)

    df = _fill_gaps(df)

    if len(df) > 0 and not current_status(run_dir)["running"]:
        last_stop = df["stop"].iloc[-1]
        gap = pd.DataFrame(
            [
                {
                    "state": PipelineState.SEARCH_DOWN.value,
                    "start": last_stop,
                    "stop": gps_now(),
                    "ifos_ready": "",
                }
            ]
        )
        df = pd.concat([df, gap], ignore_index=True)

    if start_time is not None:
        df = _window(df, start=start_time)
    return df.reset_index(drop=True)


def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.to_dict("records")
    filled = []
    for row, nxt in zip(rows, rows[1:], strict=False):
        filled.append(row)
        if nxt["start"] > row["stop"]:
            gap = pd.DataFrame(
                [
                    {
                        "state": PipelineState.SEARCH_DOWN.value,
                        "start": row["stop"],
                        "stop": nxt["start"],
                        "ifos_ready": "",
                    }
                ]
            )
            filled.append(gap.iloc[0].to_dict())
    filled.extend(rows[-1:])
    return pd.DataFrame(filled, columns=list(df.columns))


def _window(
    df: pd.DataFrame,
    start: float | None = None,
    end: float | None = None,
) -> pd.DataFrame:
    """Clamp segments to a window, dropping those outside it."""
    if start is not None:
        df = df[df["stop"] > start].copy()
        df["start"] = df["start"].clip(lower=start)
    if end is not None:
        df = df[df["start"] < end].copy()
        df["stop"] = df["stop"].clip(upper=end)
    return df


def compute_duty_cycle(
    df: pd.DataFrame,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    """
    Summarize how the search spent its time.

    The duty cycle is the fraction of analyzable time we actually
    analyzed. Detector downtime is excluded from the denominator,
    while our own downtime and filter warm-up count against us.

    Args:
        df: Segments from `load_segments`
        start: GPS time to start the window at, defaulting to the
            first segment
        end: GPS time to end the window at, defaulting to the last

    Returns:
        Seconds `elapsed`, `livetime` analyzed and `unknown` spent
        down, plus the `duty_cycle` and `uptime` fractions, which are
        None if there was no analyzable time.
    """
    df = _window(df, start, end)
    if not len(df):
        return {
            "elapsed": 0.0,
            "livetime": 0.0,
            "unknown": 0.0,
            "duty_cycle": None,
            "uptime": None,
        }

    durations = (df["stop"] - df["start"]).clip(lower=0)
    by_state = durations.groupby(df["state"]).sum()
    elapsed = df["stop"].max() - df["start"].min()

    livetime = by_state.get(PipelineState.ANALYZING, 0)
    detector_downtime = sum(by_state.get(s, 0) for s in DETECTOR_FAULT_STATES)
    analyzable = elapsed - detector_downtime
    search_downtime = by_state.get(PipelineState.SEARCH_DOWN, 0.0)

    return {
        "elapsed": elapsed,
        "livetime": livetime,
        "search_downtime": search_downtime,
        "duty_cycle": livetime / analyzable if analyzable > 0 else None,
        "uptime": 1 - search_downtime / elapsed if elapsed > 0 else None,
    }


def longest_downtimes(
    df: pd.DataFrame, n: int = 10, min_duration: float = 60.0
) -> pd.DataFrame:
    """
    The `n` longest stretches of time we were responsible for losing,
    longest first. Stretches shorter than `min_duration` are left out.
    """
    downtime = df[df["state"].isin(SEARCH_FAULT_STATES)].copy()
    downtime["duration"] = (downtime["stop"] - downtime["start"]).clip(lower=0)
    downtime = downtime[downtime["duration"] >= min_duration]
    downtime = downtime.sort_values("duration", ascending=False)
    return downtime.head(n).reset_index(drop=True)
