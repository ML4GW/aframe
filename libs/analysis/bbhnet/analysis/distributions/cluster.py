import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import h5py
import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class ClusterDistribution(Distribution):
    """
    Distribution representing a clustering of sampled
    points. The result of clustering is that no two events will be within
    t_clust / 2 of eachother.

    Args:
        t_clust: The length of the clustering window
    """

    t_clust: float

    def __post_init__(self) -> None:
        super().__post_init__()

    def _load(self, path: Path):
        """Load distribution information from an HDF5 file"""
        with h5py.File(path, "r") as f:
            self.events = f["events"][:]
            self.event_times = f["event_times"][:]
            self.shifts = f["shifts"][:]
            self.Tb = f["Tb"][()]

    def write(self, path: Path):
        """Write the distribution's data to an HDF5 file"""
        with h5py.File(path, "w") as f:
            f["events"] = self.events
            f["Tb"] = self.Tb
            f["event_times"] = self.event_times
            f["shifts"] = self.shifts
            f.attrs.update({"t_clust": self.t_clust})

    @classmethod
    def from_file(cls, dataset: str, ifos: Iterable[str], path: Path):
        """Create a new distribution with data loaded from an HDF5 file"""
        with h5py.File(path, "r") as f:
            t_clust = f.attrs["t_clust"]

        obj = cls(dataset, ifos, t_clust)
        obj._load(path)
        return obj

    def update(self, x: np.ndarray, t: np.ndarray, shifts: np.ndarray):
        """
        Update the histogram using the values from `x`,
        and update the background time `Tb` using the
        values from `t`
        """

        # update background time
        self.Tb += t[-1] - t[0] + t[1] - t[0]

        remove_indices = []

        # infer number of samples in half window
        sample_rate = 1 / (t[1] - t[0])
        half_window_size = int((self.t_clust / 2) * sample_rate)

        # for each sample, if there
        # is another sample within a half window
        # length with a greater output,
        # delete the sample
        for i in range(len(t)):

            left = max(0, i - half_window_size)
            right = min(len(t), i + half_window_size) + 1

            if any(x[left:right] > x[i]):
                remove_indices.append(i)

        events = np.delete(x, remove_indices)
        times = np.delete(t, remove_indices)

        self.events = np.append(self.events, events)
        self.event_times = np.append(self.event_times, times)

        shifts = np.repeat([shifts], len(events), axis=0)
        self.shifts = np.append(self.shifts, shifts, axis=0)

    def nb(self, threshold: Union[float, np.ndarray]):
        """
        Counts the number of events above the indicated
        `threshold`
        """

        if isinstance(threshold, np.ndarray):
            nb = [np.sum(self.events >= thresh) for thresh in threshold]
        else:
            nb = np.sum(self.events >= threshold)

        logging.debug(
            "Threshold {} has {} events greater than it "
            "in distribution {}".format(threshold, nb, self)
        )
        return np.array(nb)
