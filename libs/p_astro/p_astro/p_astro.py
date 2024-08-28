from p_astro.background import BackgroundModel
from p_astro.foreground import ForegroundModel


class Pastro:
    def __init__(
        self,
        foreground_model: ForegroundModel,
        background_model: BackgroundModel,
    ):
        self.foreground_model = foreground_model
        self.background_model = background_model

    def __call__(self, stats):
        background_rate = self.background_model(stats)
        foreground_rate = self.foreground_model(stats)
        return foreground_rate / (foreground_rate + background_rate)
