import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from bokeh.models import MultiChoice
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from plots.pages import Page
from plots.vetos import VetoParser


def normalize_path(path):
    path = Path(path)
    if not path.is_absolute():
        return Path(__file__).resolve().parent / path
    return path


VETO_DEFINER_FILE = normalize_path("./vetos/H1L1-HOFT_C01_O3_CBC.xml")
GATE_PATHS = {
    "H1": normalize_path("./vetos/H1-O3_GATES_1238166018-31197600.txt"),
    "L1": normalize_path("./vetos/L1-O3_GATES_1238166018-31197600.txt"),
}


class DataManager:
    """
    Class for managing data, including applying vetos
    """

    def __init__(self, results_dir: Path, data_dir: Path, pages: list[Page]):
        self.pages = pages
        # load results and data from the run we're visualizing
        infer_dir = results_dir / "infer" / "1year"
        rejected = data_dir / "rejected-parameters.hdf5"
        self.response_set = data_dir / "waveforms.hdf5"

        logging.info(
            "Reading in background, foreground and rejected parameters"
        )
        self.background = EventSet.read(infer_dir / "background.hdf5")
        self.foreground = RecoveredInjectionSet.read(
            infer_dir / "foreground.hdf5"
        )
        self.rejected_params = InjectionParameterSet.read(rejected)
        logging.info("Data loaded")

        # move injection masses to source frame
        # TODO: this should be done in the ledger
        for obj in [self.foreground, self.rejected_params]:
            for i in range(2):
                attr = f"mass_{i + 1}"
                value = getattr(obj, attr)
                setattr(obj, attr, value / (1 + obj.redshift))

        # create copies of the background and foreground
        # for applying vetoes
        self._background = deepcopy(self.background)
        self._foreground = deepcopy(self.foreground)

        self.veto_parser = VetoParser(
            VETO_DEFINER_FILE,
            GATE_PATHS,
            self._background.detection_time.min(),
            self._background.detection_time.max(),
            self.ifos,
        )

    @property
    def veto_options(self):
        return ["CAT1", "CAT2", "CAT3", "GATES"]

    def get_veto_selecter(self):
        return MultiChoice(
            title="Applied Vetos", value=[], options=self.veto_options
        )

    def calculate_veto_masks(self):
        self.vetoes = {}
        for label in self.veto_options:
            logging.info(f"Calculating veto mask for {label}")
            vetos = self.veto_parser.get_vetoes(label)
            veto_mask = False
            for i, ifo in enumerate(self.ifos):
                segments = vetos[ifo]

                # apply vetoes of background
                _, mask = self.background.apply_vetos(
                    segments, i, inplace=False
                )

                # mark a background event as vetoed
                # if it is vetoed in any if
                veto_mask |= mask

            self.vetoes[label] = veto_mask

        logging.info("Veto masks calculated")
        self.veto_mask = np.zeros_like(mask, dtype=bool)

    def update_vetos(self, attr, old, new):
        if not new:
            # no vetoes selected, so mark all background
            # events as not-vetoed
            self.veto_mask = np.zeros_like(self.veto_mask, dtype=bool)
        else:
            # mark a background event as vetoed if any
            # of the currently selected labels veto it
            mask = False
            for label in new:
                mask |= self.vetoes[label]
            self.veto_mask = mask

        # update vetos in our data object
        background = self._background[~self.veto_mask]
        foreground = self._foreground[~self.veto_mask]

        # update pages with latest background and foreground
        for page in self.pages:
            page.update(background, foreground)
