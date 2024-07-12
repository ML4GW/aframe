import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from bokeh.models import MultiChoice
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from plots.vetos import VetoParser


def chirp_mass(m1, m2):
    """Calculate chirp mass from component masses"""
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)


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

    def __init__(self, results_dir: Path, data_dir: Path, ifos: list[str]):
        self.logger = logging.getLogger("vizapp")
        self.ifos = ifos
        # load results and data from the run we're visualizing
        infer_dir = results_dir / "infer" / "1year"
        rejected = data_dir / "rejected-parameters.hdf5"
        self.response_set = data_dir / "waveforms.hdf5"

        self.logger.info(
            "Reading in background, foreground and rejected parameters"
        )
        self.background = EventSet.read(infer_dir / "background.hdf5")
        self.foreground = RecoveredInjectionSet.read(
            infer_dir / "foreground.hdf5"
        )
        self.rejected_params = InjectionParameterSet.read(rejected)
        self.logger.info("Data loaded")

        # add source frame masses
        # TODO: this should be done in the ledger / ts waveforms
        for obj in [self.foreground, self.rejected_params]:
            for i in range(2):
                attr = f"mass_{i + 1}"
                value = getattr(obj, attr)
                attr = f"{attr}_source"
                setattr(obj, attr, value / (1 + obj.redshift))
                setattr(obj, "chirp_mass", chirp_mass(obj.mass_1, obj.mass_2))

        # create copies of the background and foreground
        # for applying vetos
        self._background = deepcopy(self.background)
        self._foreground = deepcopy(self.foreground)

        self.veto_parser = VetoParser(
            VETO_DEFINER_FILE,
            GATE_PATHS,
            self._background.detection_time.min(),
            self._background.detection_time.max(),
            self.ifos,
        )
        self.calculate_veto_masks()

    @property
    def veto_options(self):
        return [
            "GATES",
            "CAT1",
            "CAT2",
            "CAT3",
        ]

    def get_veto_selecter(self):
        return MultiChoice(
            title="Applied Vetos", value=[], options=self.veto_options
        )

    def calculate_veto_masks(self):
        self.vetos = {}
        for label in self.veto_options:
            self.logger.info(f"Calculating veto mask for {label}")
            vetos = self.veto_parser.get_vetos(label)
            veto_mask = False
            for i, ifo in enumerate(self.ifos):
                segments = vetos[ifo]
                # apply vetos to background
                _, mask = self.background.apply_vetos(
                    segments, i, inplace=False
                )

                # mark a background event as vetoed
                # if it is vetoed in any ifo
                veto_mask |= mask

            self.vetos[label] = veto_mask

        self.logger.info("Veto masks calculated")

        self.veto_mask = np.zeros_like(mask, dtype=bool)

    def update_vetos(self, attr, old, new):
        if not new:
            # no vetos selected, so mark all background
            # events as not-vetoed
            self.veto_mask = np.zeros_like(self.veto_mask, dtype=bool)
        else:
            # mark a background event as vetoed if any
            # of the currently selected labels veto it
            mask = False
            for label in new:
                mask |= self.vetos[label]
            self.veto_mask = mask

        # update vetos in our data object
        background = self._background[~self.veto_mask]

        # TODO: apply foreground vetos
        foreground = self._foreground
        return background, foreground
