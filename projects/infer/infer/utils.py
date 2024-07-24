from pathlib import Path
from typing import List

import h5py


def get_shifts(files: List[Path]):
    shifts = []
    for f in files:
        with h5py.File(f) as f:
            shift = f["parameters"]["shift"][0]
            shifts.append(shift)
    return shifts
