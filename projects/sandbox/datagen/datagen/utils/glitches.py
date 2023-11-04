import re
from pathlib import Path

# omicron trigger files
re_trigger_fname = re.compile(
    r"(?P<ifo>[A-Za-z]1)-(?P<name>.+)-(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*).h5$"  # noqa
)


def get_state_flag(state_flag: str):
    if state_flag == "DATA":
        return "DCS-ANALYSIS_READY_C01:1"
    return state_flag


def get_channel(channel: str):
    if channel == "OPEN":
        return "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01"
    return channel


def parse_omicron_fname(f: Path):
    match = re_trigger_fname.search(f.name)
    if match is not None:
        t0, length = float(match.group("t0")), float(match.group("length"))
        t0, length = intify(t0), intify(length)
        return t0, length
    raise ValueError(f"Could not parse trigger file name {f.name}")


def intify(x: float):
    return int(x) if int(x) == x else x


def handle_future(future):
    try:
        x = future.result()
    except Exception as e:
        raise e
    return x
