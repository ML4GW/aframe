from typing import TYPE_CHECKING

from plots.vizapp.pages.page import Page

from .sv import SensitiveVolumePlot

if TYPE_CHECKING:
    from ledger.events import EventSet, RecoveredInjectionSet


class Summary(Page):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sv = SensitiveVolumePlot(
            self.app.data_manager.background,
            self.app.data_manager.foreground,
            self.app.data_manager.rejected_params,
            self.app.mass_combos,
            self.app.source_prior,
        )

    def get_layout(self):
        return self.sv.get_layout()

    def update(
        self, background: "EventSet", foreground: "RecoveredInjectionSet"
    ):
        self.sv.update(background, foreground)
