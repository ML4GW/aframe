from typing import List, Tuple

import h5py
import numpy as np


def discrete_distribution(
    fnames: List[str],
    min_val: float,
    max_val: float,
    num_bins: int,
    group: str = "filtered",
) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(min_val, max_val, num_bins)
    hist = np.zeros((num_bins,))
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            try:
                values = f[group][:]
            except KeyError:
                raise ValueError(
                    f"HDF archive '{fname}' has no group '{group}'"
                )
        counts, _ = np.histogram(values, bins)
        hist += counts
    return hist, (bins[:-1] + bins[1:]) / 2
