import logging
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
from gwosc import datasets
from gwpy.segments import DataQualityDict

# TODO: gating should really be applied directly to the strain


VETO_CATEGORIES = Literal["CAT1", "CAT2", "CAT3", "GATES", "CATALOG"]


def gates_to_veto_segments(path: Path):
    """Naively convert gate files to vetos segments"""
    gates = np.loadtxt(path)
    centers = gates[:, 0]
    windows = gates[:, 1]
    tapers = gates[:, 2] + 0.375

    vetos = np.array(
        [
            [center - window - taper, center + window + taper]
            for center, window, taper in zip(centers, windows, tapers)
        ]
    )

    return vetos


def get_catalog_vetos(start: float, stop: float, delta: float = 1.0):
    events = datasets.query_events(
        select=[f"gps-time >= {start}", f"gps-time <= {stop}"]
    )
    times = np.array([datasets.event_gps(event) for event in events])
    vetos = np.column_stack([times - delta, times + delta])
    return vetos


class VetoParser:
    def __init__(
        self,
        veto_definer_file: Path,
        gate_paths: Dict[str, Path],
        start: float,
        stop: float,
        ifos: List[str],
    ):
        self.logger = logging.getLogger("vizapp")
        self.vetos = DataQualityDict.from_veto_definer_file(veto_definer_file)
        self.logger.info("Populating vetos")
        self.vetos.populate(segments=[[start, stop]], verbose=True)
        self.logger.info("Vetos populated")
        self.gate_paths = gate_paths
        self.ifos = ifos
        self.veto_cache = {}

    def get_vetos(self, category: str):
        vetos = {}

        for ifo in self.ifos:
            if category == "GATES":
                ifo_vetos = gates_to_veto_segments(self.gate_paths[ifo])
            else:
                cat_number = int(category[-1])
                ifo_vetos = DataQualityDict(
                    {
                        k: v
                        for k, v in self.vetos.items()
                        if v.ifo == ifo and v.category == cat_number
                    }
                )
                ifo_vetos = ifo_vetos.union().active

            vetos[ifo] = np.array(ifo_vetos)

        return vetos
