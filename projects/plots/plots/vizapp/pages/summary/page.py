from typing import TYPE_CHECKING

from plots.core.constants import SECONDS_PER_MONTH, SECONDS_PER_YEAR
from plots.core.data import AnalysisData
from plots.core.gwtc3 import main as gwtc3_pipeline_sv
from plots.core.sv import (
    SensitiveVolumePlot,
    comparisons_from_gwtc3_curves,
    compute_sensitive_volume,
)
from plots.vizapp.pages.page import Page

if TYPE_CHECKING:
    from ledger.events import EventSet, RecoveredInjectionSet

VIZAPP_MAX_FAR = 100 * SECONDS_PER_YEAR / SECONDS_PER_MONTH


class Summary(Page):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_manager = self.app.data_manager
        data = AnalysisData(
            data_manager.background,
            data_manager.foreground,
            data_manager.rejected_params,
        )
        result = compute_sensitive_volume(
            data,
            mass_combos=self.app.mass_combos,
            source_prior=self.app.source_prior,
            max_far=VIZAPP_MAX_FAR,
        )

        gwtc3_sv, gwtc3_err = gwtc3_pipeline_sv(
            mass_combos=self.app.mass_combos,
            detection_criterion="far",
            detection_thresholds=result.fars,
            output_dir=self.app.results_dir,
        )
        comparisons = comparisons_from_gwtc3_curves(
            gwtc3_sv, gwtc3_err, self.app.mass_combos
        )
        self.sv = SensitiveVolumePlot(result, comparisons)

    def get_layout(self):
        return self.sv.layout()

    def update(
        self, background: "EventSet", foreground: "RecoveredInjectionSet"
    ):
        data = AnalysisData(
            background,
            foreground,
            self.app.data_manager.rejected_params,
        )
        result = compute_sensitive_volume(
            data,
            mass_combos=self.app.mass_combos,
            source_prior=self.app.source_prior,
            max_far=VIZAPP_MAX_FAR,
        )
        self.sv.update(result)
