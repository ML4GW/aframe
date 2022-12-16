from dataclasses import dataclass

import numpy as np

from bbhnet.analysis.integrators import boxcar_filter


class Normalizer:
    def fit(self, y: np.ndarray):
        # TODO: raise not implemented? What if
        # a child class doesn't need to fit?
        return

    def __call__(self, integrated: np.ndarray, window_size: int) -> np.ndarray:
        # TODO: raise NotImplementedError here?
        return integrated


@dataclass
class GaussianNormalizer:
    norm_size: int

    def __post_init__(self):
        if int(self.norm_size) != self.norm_size:
            raise ValueError(f"'norm_size' {self.norm_size} not an integer")
        self.norm_size = int(self.norm_size)
        self.shifts = None
        self.scales = None

    def fit(self, y: np.ndarray):
        # compute the mean and standard deviation of
        # the last `norm_seconds` seconds of data
        # compute the standard deviation by the
        # sigma^2 = E[x^2] - E^2[x] trick
        shifts = boxcar_filter(y, self.norm_size)
        sqs = boxcar_filter(y**2, self.norm_size)
        scales = np.sqrt(sqs - shifts**2)
        if (scales == 0).any():
            raise ValueError("Encountered 0s in scale parameter")

        self.shifts = shifts
        self.scales = scales

    def __call__(self, integrated: np.ndarray, window_size: int) -> np.ndarray:
        if self.shifts is None or self.scales is None:
            raise ValueError("GaussianNormalizer hasn't been fit")
        if len(integrated) != len(self.shifts):
            raise ValueError(
                "Can't normalize timeseries of length {} "
                "with GaussianNormalizer fit to shape {}".format(
                    len(integrated), len(self.shifts)
                )
            )

        # get rid of the first norm_seconds worth of data
        # since there's nothing to normalize by
        # also include an offset by the window size so that
        # the average within a window is being compared to
        # the mean and standard deviation of samples entirely
        # _before_ the given window
        shifts = self.shifts[self.norm_size : -window_size]
        scales = self.scales[self.norm_size : -window_size]
        integrated = integrated[self.norm_size + window_size :]
        return (integrated - shifts) / scales
