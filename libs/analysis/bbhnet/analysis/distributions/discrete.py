import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class DiscreteDistribution(Distribution):
    """
    Distribution representing a discretization of sampled
    points into evenly spaced bins between a minimum and
    maximum value.

    Args:
        minimum: The lowest left edge of the distribution's bins
        maximum: The highest right edge (exclusive) of the distribution's bins
        num_bins: The number of bins to use for discretization
        clip:
            Whether to ignore values outside `[minmum, maximum)` or
            add these counts to the leftmost or rightmost bins
    """

    minimum: float
    maximum: float
    num_bins: float
    clip: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.bins = np.linspace(self.minimum, self.maximum, self.num_bins + 1)
        self.histogram = np.zeros((self.num_bins,))

    # TODO: at this point, just use pickle?
    def write(self, path: Path):
        """Write the distribution's data to an HDF5 file"""
        with h5py.File(path, "w") as f:
            f["bins"] = self.bins
            f["histogram"] = self.histogram
            f["Tb"] = self.Tb

    def load(self, path: Path):
        """Load distribution information from an HDF5 file"""
        with h5py.File(path, "r") as f:
            self.bins = f["bins"][:]
            self.histogram = f["histogram"][:]
            self.Tb = f["Tb"]

    @classmethod
    def from_file(cls, dataset: str, path: Path, clip: bool = False):
        """Create a new distribution with data loaded from an HDF5 file"""
        with h5py.File(path, "r") as f:
            bins = f["bins"][:]
        obj = cls(dataset, bins.min(), bins.max(), len(bins) - 1, clip)
        obj.load(path)
        return obj

    @property
    def bin_centers(self):
        return (self.bins[:-1] + self.bins[1:]) / 2

    def nb(self, threshold: float):
        """
        Counts the number of events above the indicated
        `threshold` by summing over bins whose left edge
        is greater than or equal to the `threshold`. Note
        that this means that samples from the bin _containing_
        the `threshold` will _not_ be included (which might
        not be optimal, open to other ideas here).
        """

        # TODO: is this the best way to do this check?
        # TODO: should we actually be building the mask
        # by looking at bins whose right edge is less
        # than the threshold, to include samples from
        # the same bin?
        if isinstance(threshold, np.ndarray):
            bins = np.repeat(self.bins[:-1, None], len(threshold), axis=1)
            hist = np.repeat(self.histogram[:, None], len(threshold), axis=1)
            mask = bins >= threshold
            nb = (hist * mask).sum(axis=0)
        else:
            nb = self.histogram[self.bins[:-1] >= threshold].sum(axis=0)

        logging.debug(
            "Threshold {} has {} events greater than it "
            "in distribution {}".format(threshold, nb, self)
        )
        return nb

    def update(self, x: np.ndarray, t: np.ndarray):
        """
        Update the histogram using the values from `x`,
        and update the background time `Tb` using the
        values from `t`
        """

        counts, _ = np.histogram(x, self.bins)
        if counts.sum() < len(x) and not self.clip:
            counts[0] += (x < self.bins[0]).sum()
            counts[-1] += (x >= self.bins[-1]).sum()

        self.histogram += counts
        self.Tb += t[-1] - t[0] + t[1] - t[0]
