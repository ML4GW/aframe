import numpy as np

from p_astro.background import BackgroundModel
from p_astro.foreground import ForegroundModel


class Pastro:
    """
    p_astro model incorporating foreground and background distribution models.

    Args:
        foreground_model: Model for the astrophysical foreground events
        background_model: Model for the background (noise) events
    """

    def __init__(
        self,
        foreground_model: ForegroundModel,
        background_model: BackgroundModel,
    ):
        self.foreground_model = foreground_model
        self.background_model = background_model

    def __call__(self, stats: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the probability that an event with the given detection
        statistic is of astrophysical origin.

        Args:
            stats: Detection statistic(s) at which to evaluate the
                astrophysical probability

        Returns:
            Astrophysical probability evaluated at the input detection
            statistic(s)
        """
        background_rate = self.background_model(stats)
        foreground_rate = self.foreground_model(stats)
        return foreground_rate / (foreground_rate + background_rate)
