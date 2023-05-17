from vizapp.pages.page import Page
from vizapp.pages.performance_summary.sensitive_volume import (
    SensitiveVolumePlot,
)


class PerformanceSummaryPage(Page):
    def __init__(self, app):
        super().__init__(app)
        self.sensitive_volume = SensitiveVolumePlot(self)

    def get_layout(self):
        return self.sensitive_volume.get_layout()

    def update(self):
        self.sensitive_volume.update()
