import json
from pathlib import Path

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


def segment_dir(run_dir: Path) -> Path:
    return run_dir / "output" / "segments"


def read_heartbeat(run_dir: Path) -> dict | None:
    """
    Read the search's heartbeat file, or None if there isn't one.
    """
    try:
        with open(segment_dir(run_dir) / CURRENT_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def current_status(run_dir: Path) -> str | None:
    """
    The state the search is currently in, or None if its heartbeat
    has gone stale.
    """
    heartbeat = read_heartbeat(run_dir)
    if heartbeat is None:
        return None
    if gps_now() - heartbeat["heartbeat"] > STALE_SECONDS:
        return None
    return heartbeat["state"]


def load_segments(
    run_dir: Path, start_time: float | None = None
) -> pd.DataFrame:
    """
    Load the search's segment record, filling in the stretches it
    couldn't write itself.

    The search appends a row to `segments.txt` whenever its state
    changes, with GPS start and stop times and one digit per
    interferometer for whether it was analysis-ready (blank during
    startup):

        state,start,stop,ifos_ready
        startup,1473875405.00000,1473875465.00000,
        warmup,1473875465.00000,1473875525.00000,111
        analyzing,1473875525.00000,1473879900.00000,111
        startup,1473880000.00000,1473880075.00000,
        not_ready,1473880075.00000,1473890075.00000,011

    The segment it's currently in lives in `current.json` until it
    ends.

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

    if len(df) > 0 and current_status(run_dir) is None:
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
        df:
            Segments from `load_segments`, where the gap in the
            example there, from a crash, gets filled in:

                         state            start             stop ifos_ready
                0      startup 1473875405.00000 1473875465.00000
                1       warmup 1473875465.00000 1473875525.00000        111
                2    analyzing 1473875525.00000 1473879900.00000        111
                3  search_down 1473879900.00000 1473880000.00000
                4      startup 1473880000.00000 1473880075.00000
                5    not_ready 1473880075.00000 1473890075.00000        011

        start: GPS time to start the window at, defaulting to the
            first segment
        end: GPS time to end the window at, defaulting to the last

    Returns:
        Seconds `elapsed`, `analyzable`, `livetime` analyzed and
        `search_downtime` spent down, plus the `duty_cycle` and
        `uptime` fractions, which are None if there was no analyzable
        time.
    """
    df = _window(df, start, end)
    if not len(df):
        return {
            "elapsed": 0.0,
            "analyzable": 0.0,
            "livetime": 0.0,
            "search_downtime": 0.0,
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
        "analyzable": analyzable,
        "livetime": livetime,
        "search_downtime": search_downtime,
        "duty_cycle": livetime / analyzable if analyzable > 0 else None,
        "uptime": 1 - search_downtime / elapsed if elapsed > 0 else None,
    }


def downtime_breakdown(
    df: pd.DataFrame, analyzable: float | None
) -> pd.DataFrame:
    """
    What each search-side fault cost us, worst first. `cost` is the
    fraction of analyzable time lost to that state, so the costs sum
    to one minus the duty cycle.

    Args:
        df: Segments from `load_segments`
        analyzable: Seconds of analyzable time to report costs against
    """
    faults = df[df["state"].isin(SEARCH_FAULT_STATES)]
    durations = (faults["stop"] - faults["start"]).clip(lower=0)
    breakdown = durations.groupby(faults["state"]).agg(["sum", "count", "max"])
    breakdown.columns = ["total", "occurrences", "longest"]
    if analyzable:
        breakdown["cost"] = breakdown["total"] / analyzable
    else:
        breakdown["cost"] = None
    return breakdown.sort_values("total", ascending=False).reset_index()
