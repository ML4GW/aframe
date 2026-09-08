import logging
from copy import deepcopy
from pathlib import Path

from bokeh.models import MultiChoice

from plots.core.data import AnalysisData
from plots.vetos import VETO_CATEGORIES
from plots.vetos.masks import (
    combine_masks,
    compute_veto_masks,
    load_or_fetch_segments,
)


class DataManager:
    """
    Class for managing data, including applying vetos
    """

    def __init__(
        self,
        results_dir: Path,
        waveforms_dir: Path,
        ifos: list[str],
        vetos: list[VETO_CATEGORIES] | None = None,
    ):
        self.logger = logging.getLogger("vizapp")
        self.ifos = ifos
        self.categories = vetos
        # load results and data from the run we're visualizing
        self.response_set = waveforms_dir / "waveforms.hdf5"

        data = AnalysisData.load(
            background=results_dir / "background.hdf5",
            foreground=results_dir / "foreground.hdf5",
            rejected=waveforms_dir / "rejected_parameters.hdf5",
        )
        self.background = data.background
        self.foreground = data.foreground
        self.rejected_params = data.rejected
        self.logger.info("Data loaded")

        # create copies of the background and foreground
        # for applying vetos
        self._background = deepcopy(self.background)
        self._foreground = deepcopy(self.foreground)

        self.background_masks = None
        self.foreground_masks = None
        if self.categories:
            start = self._background.detection_time.min()
            stop = self._background.detection_time.max()
            segments = load_or_fetch_segments(
                self.categories, self.ifos, start, stop
            )
            self.background_masks = compute_veto_masks(
                self._background, self.categories, self.ifos, segments
            )
            self.foreground_masks = compute_veto_masks(
                self._foreground, self.categories, self.ifos, segments
            )

    def get_veto_selecter(self):
        options = self.categories if self.categories else ["N/A"]
        return MultiChoice(title="Applied Vetos", value=[], options=options)

    def update_vetos(self, attr, old, new):
        if not self.categories:
            return self._background, self._foreground

        back_mask = combine_masks(self.background_masks, new)
        fore_mask = combine_masks(self.foreground_masks, new)
        background = self._background[~back_mask]
        foreground = self._foreground[~fore_mask]
        return background, foreground
