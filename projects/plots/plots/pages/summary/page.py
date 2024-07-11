from plots.pages.page import Page

from .sv import SensitiveVolumePlot


class Summary(Page):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        source_prior, _ = self.app.data.source_prior()
        self.sv = SensitiveVolumePlot(
            self.app.data.background,
            self.app.data.foreground,
            self.app.data.rejected_params,
            self.app.data.mass_combos,
            source_prior,
        )

    def get_layout(self):
        return self.sv.get_layout()

    def update(self):
        self.sv.update()
