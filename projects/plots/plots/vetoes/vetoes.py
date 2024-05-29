from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from gwosc import datasets
from gwpy.segments import DataQualityDict

# TODO: gating should really be applied directly to the strain


CATEGORIES = ["CAT1", "CAT2", "CAT3", "GATES"]


def gates_to_veto_segments(path: Path):
    """Naively convert gate files to vetoes segments"""
    gates = np.loadtxt(path)
    centers = gates[:, 0]
    windows = gates[:, 1]
    tapers = gates[:, 2] + 0.375

    vetoes = np.array(
        [
            [center - window - taper, center + window + taper]
            for center, window, taper in zip(centers, windows, tapers)
        ]
    )

    return vetoes


def get_catalog_vetoes(start: float, stop: float, delta: float = 1.0):
    events = datasets.query_events(
        select=[f"gps-time >= {start}", f"gps-time <= {stop}"]
    )
    times = np.array([datasets.event_gps(event) for event in events])
    vetoes = np.column_stack([times - delta, times + delta])
    return vetoes


@dataclass
class VetoParser:
    veto_definer_file: Path
    gate_paths: Dict[str, Path]
    start: float
    stop: float
    ifos: List[str]

    def __post_init__(self):
        self.vetoes = DataQualityDict.from_veto_definer_file(
            self.veto_definer_file
        )
        self.vetoes.populate(segments=[[self.start, self.stop]], verbose=False)
        self.veto_cache = {}

    def get_vetoes(self, category: str):
        vetoes = {}

        for ifo in self.ifos:
            if category == "GATES":
                ifo_vetoes = gates_to_veto_segments(self.gate_paths[ifo])
            else:
                cat_number = int(category[-1])
                ifo_vetoes = DataQualityDict(
                    {
                        k: v
                        for k, v in self.vetoes.items()
                        if v.ifo == ifo and v.category == cat_number
                    }
                )
                ifo_vetoes = ifo_vetoes.union().active

            vetoes[ifo] = np.array(ifo_vetoes)

        return vetoes
