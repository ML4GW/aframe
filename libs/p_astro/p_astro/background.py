import numpy as np
from ledger.events import SECONDS_IN_YEAR, EventSet
from numpy.polynomial import Polynomial
from scipy.stats import gaussian_kde


class BackgroundModel:
    def __init__(self, background: EventSet):
        self.background = background
        # fit should set all attributes
        # necessary to construct the model
        self.fit()

    def fit(self):
        raise NotImplementedError

    def __call__(self, stats):
        raise NotImplementedError


class KdeAndPolynomialBackground(BackgroundModel):
    def __init__(self, *args, split=None, downsampled_points=None, **kwargs):
        self.split = split
        self.downsampled_points = downsampled_points
        super().__init__(*args, **kwargs)

    @property
    def scale_factor(self):
        return len(self.background) * SECONDS_IN_YEAR / self.background.Tb

    def fit(self):
        # fit gaussian kde to the background
        kde = gaussian_kde(self.background.detection_statistic)

        # downsample the background if requested
        downsampled_points = self.downsampled_points or len(self.background)
        downsampled_factor = len(self.background) // downsampled_points
        downsampled = self.background.detection_statistic[::downsampled_factor]
        downsampled_kde = gaussian_kde(downsampled)

        # Estimate the peak of the distribution
        samples = np.linspace(
            self.background.detection_statistic.min(),
            self.background.detection_statistic.max(),
            100,
        )
        pdf = downsampled_kde(samples)
        peak_idx = np.argmax(pdf)

        if self.split is not None:
            stop = np.argmin(np.abs(samples - self.split))
            threshold_pdf_value = pdf[stop]
            start = (
                np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
            )
        else:
            # Determine the range of values to use for fitting
            # a line to a portion of the pdf.
            # Roughly, we have too few samples to properly
            # estimate the KDE once the pdf drops below 1/sqrt(N)
            threshold_pdf_value = 1 / np.sqrt(len(downsampled))
            start = (
                np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
            )
            stop = np.argmin(pdf[peak_idx:] > threshold_pdf_value) + peak_idx

        # Fit a line to the log pdf of the region prior to the split point
        polynomial_samples = samples[start:stop]

        # set all attributes necessary to construct the model
        # so that the object can be pickled / saved to disk
        self.kde = kde
        self.polynomial = Polynomial.fit(
            polynomial_samples, np.log(pdf[start:stop]), 1
        )
        self.threshold_statistic = samples[stop]

    def __call__(self, stats):
        return (
            np.piecewise(
                stats,
                [
                    stats < self.threshold_statistic,
                    stats >= self.threshold_statistic,
                ],
                [self.kde, lambda x: np.exp(self.polynomial(x))],
            )
            * self.scale_factor
        )
