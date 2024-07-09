from pathlib import Path
import re

t0_pattern = re.compile(r"[0-9]{10}(\.[0-9])?(?=-)")
dur_pattern = re.compile("[0-9]{2,8}(?=.hdf5)")


def get_strain_fname(data_dir: Path, time: float):
    # loop over files in data directory,
    # extracting t0 and duraiton via regexes
    for fname in data_dir.iterdir():
        try:
            t0 = float(t0_pattern.search(fname.name).group(0))
            dur = float(dur_pattern.search(fname.name).group(0))
        except AttributeError:
            continue

        # break if this file contains event of interest
        if t0 <= time < (t0 + dur):
            break
    # if no file found, raise error
    else:
        raise ValueError(
            "No file containing event time {} in "
            "data directory {}".format(time, data_dir)
        )
    return fname, t0, dur