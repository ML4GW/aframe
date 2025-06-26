import psutil
from pathlib import Path
from gwpy.time import tconvert
from datetime import datetime, timezone
import pytz


def get_log_files(log_dir: Path, start_time: float) -> list:
    """
    Get log files that have data newer than start_time

    Args:
        log_dir: Directory containing log files.
        start_time: The earliest timestamp to consider for logs.

    Returns:
        List of log file paths.
    """
    all_log_files = sorted(log_dir.glob("**/*.log*"))
    modified_times = [f.stat().st_mtime for f in all_log_files]
    modified_datetimes = [
        datetime.fromtimestamp(mtime, tz=pytz.timezone("US/Pacific"))
        for mtime in modified_times
    ]
    modified_utc_datetimes = [
        dt.astimezone(timezone.utc) for dt in modified_datetimes
    ]
    relevant_log_files = [
        f
        for f, dt in zip(all_log_files, modified_utc_datetimes)
        if dt >= start_time
    ]
    return sorted(relevant_log_files)


def get_timestamp_from_log_statement(log_statement: str) -> float:
    datetime_string = log_statement[:23]
    datetime_string = datetime_string.replace(",", ".")
    return datetime.fromisoformat(datetime_string).replace(tzinfo=timezone.utc)


def get_tb_from_log_text(log_text: list[str], start_time: float) -> float:
    exit_lines = [
        "H1 exiting analysis ready mode\n",
        "L1 exiting analysis ready mode\n",
    ]
    reset_line = "is ready again, resetting states\n"
    live_segments = []

    # Cut out any lines before the start time
    for i, line in enumerate(log_text):
        if not line.startswith("202"):
            continue
        if get_timestamp_from_log_statement(line) >= start_time:
            log_text = log_text[i:]
            break

    # Have the first segment start at the timestamp of the first line
    # This isn't exactly correct for logs that include the start-up
    # details, but it's needed for logs that are started at midnight
    # and it shouldn't throw the calculation off by much
    segment = [get_timestamp_from_log_statement(log_text[0])]
    for line in log_text:
        if not line.startswith("202"):
            continue
        if line.endswith(reset_line) and not segment:
            segment.append(get_timestamp_from_log_statement(line))
            continue
        if any(line.endswith(exit_line) for exit_line in exit_lines):
            if segment:
                segment.append(get_timestamp_from_log_statement(line))
                live_segments.append(segment)
            segment = []
    if segment:
        segment.append(get_timestamp_from_log_statement(line))
        live_segments.append(segment)
    return (
        sum([(stop - start).total_seconds() for start, stop in live_segments])
        if live_segments
        else 0.0
    )


def estimate_tb(run_dir: Path, start_time: float) -> float:
    log_dir = run_dir / "output" / "logs"
    start_time = tconvert(start_time).replace(tzinfo=timezone.utc)
    log_files = get_log_files(log_dir, start_time)
    if not log_files:
        return 0.0
    log_text = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            log_text.append(f.readlines())
    log_text = [line for sublist in log_text for line in sublist]
    return get_tb_from_log_text(log_text, start_time)


def pipeline_online():
    online_processes = 0
    for p in psutil.process_iter(["username", "name"]):
        if p.info["username"] == "aframe" and p.info["name"] == "online":
            online_processes += 1
    return online_processes > 1


def data_ready(run_dir: Path):
    if not pipeline_online():
        return
    log_dir = sorted((run_dir / "output" / "logs").iterdir())[-1]
    log_file = log_dir / "online.log"
    # Given the current logging, I don't think
    # there's anything more efficient than loading
    # the entire log file. The files change each
    # day, so this shouldn't ever be too bad
    with open(log_file, "r") as f:
        lines = f.readlines()

    failure_lines = [
        "H1 exiting analysis ready mode\n",
        "L1 exiting analysis ready mode\n",
        "H1 not analysis ready\n",
        "L1 not analysis ready\n",
    ]

    ready_line = "is ready again, resetting states\n"

    # Go through the lines in reverse order and
    # return the state based on whichever condition
    # we find first
    for line in lines[::-1]:
        if any(line.endswith(failure) for failure in failure_lines):
            return False
        if line.endswith(ready_line):
            return True
    return True
