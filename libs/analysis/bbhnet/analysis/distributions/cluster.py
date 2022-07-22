from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class ClusterDistribution(Distribution):
    """
    Distribution representing a clustering of sampled
    points over consecutive windows of length `t_clust`
    Args:
        t_clust: The length of the clustering window
    """

    t_clust: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events = []

    def _load(self, path: Path):
        """Load distribution information from an HDF5 file"""
        with h5py.File(path, "r") as f:
            self.events = f["events"][:]
            self.fnames = list(f["fnames"][:])
            self.Tb = f["Tb"]
            t_clust = f.attrs["t_clust"]
            if t_clust != self.t_clust:
                raise ValueError(
                    "t_clust of Distribution object {t_clust}"
                    "does not match t_clust of data in h5 file {self.t_clust}"
                )

    def write(self, path: Path):
        """Write the distribution's data to an HDF5 file"""
        with h5py.File(path, "w") as f:
            f["events"] = self.events
            f["fnames"] = list(map(str, self.fnames))
            f["Tb"] = self.Tb
            f.attrs.update({"t_clust": self.t_clust})

    @classmethod
    def from_file(cls, dataset: str, path: Path):
        """Create a new distribution with data loaded from an HDF5 file"""
        with h5py.File(path, "r") as f:
            t_clust = f.attrs["t_clust"]
        obj = cls(dataset, t_clust)
        obj._load(path)
        return obj

    def update(self, x: np.ndarray, t: np.ndarray):
        """
        Update the histogram using the values from `x`,
        and update the background time `Tb` using the
        values from `t`
        """

        # cluster values over window length
        sample_rate = t[1] - t[0]

        # samples per cluster window
        clust_size = int(sample_rate * self.t_clust)

        # take care of reshaping

        try:
            x = x.reshape((-1, clust_size))
            maxs = list(np.amax(x, axis=-1))
        except ValueError:
            extra = len(x) % clust_size
            extra_max = max(x[-extra:])

            x = x[:-extra]
            x = x.reshape((-1, clust_size))
            maxs = list(np.amax(x, axis=-1))
            maxs.append(extra_max)

        # update events and livetime
        self.events.extend(maxs)
        self.Tb += t[-1] - t[0] + t[1] - t[0]

    def nb(self, threshold: float):
        """
        Counts the number of events above the indicated
        `threshold`
        """
        nb = np.sum(np.array(self.events) >= threshold)
        return nb
