from plots.data import Data

from .sv import SensitiveVolumePlot


class Summary:
    def __init__(self, data: Data):
        self.data = data

        source_prior, _ = data.source_prior()
        self.sv = SensitiveVolumePlot(
            data.background,
            data.foreground,
            data.rejected_params,
            data.mass_combos,
            source_prior,
        )

    def get_layout(self):
        return self.sv.get_layout()

    def update(self):
        self.sv.update()
