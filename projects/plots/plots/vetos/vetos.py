import logging
from pathlib import Path
from typing import Literal

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
            for center, window, taper in zip(
                centers, windows, tapers, strict=True
            )
        ]
    )

    return vetos


# Without $DEFAULT_SEGMENT_SERVER, dqsegdb2 falls back
# to segments.ligo.org, which was shut down on 2026-09-10.
DEFAULT_SEGMENT_SERVER = "https://segments.igwn.org"


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
        gate_paths: dict[str, Path],
        start: float,
        stop: float,
        ifos: list[str],
        segment_server: str = DEFAULT_SEGMENT_SERVER,
    ):
        self.logger = logging.getLogger("vizapp")
        self.veto_definer_file = Path(veto_definer_file)
        self.vetos = DataQualityDict.from_veto_definer_file(veto_definer_file)
        self.vetos = DataQualityDict(
            {k: v for k, v in self.vetos.items() if v.ifo in ifos}
        )
        self.logger.info(
            f"Populating {len(self.vetos)} vetos from "
            f"{self.veto_definer_file.name} over [{start:.0f}, {stop:.0f})"
        )
        self.vetos.populate(
            source=segment_server, segments=[[start, stop]], verbose=True
        )
        self.logger.info("Vetos populated")
        self.gate_paths = gate_paths
        self.ifos = ifos
        self.veto_cache = {}

    @property
    def categories(self) -> list[str]:
        """The `CAT<n>` categories this definer actually defines.

        Not every run has all three: the O4 CBC definers are CAT1 only.
        """
        return [
            f"CAT{n}"
            for n in sorted({v.category for v in self.vetos.values()})
        ]

    def get_vetos(self, category: str):
        if category != "GATES" and category not in self.categories:
            raise ValueError(
                f"{self.veto_definer_file.name} defines no {category} flags, "
                f"only {self.categories}. Requesting it would fail inside "
                "gwpy when the empty flag set is unioned."
            )
        if category == "GATES" and not self.gate_paths:
            raise ValueError(f"No gate files available for {self.ifos}")

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
                # union() reduces without an initial value, so an IFO with no
                # flags in this category has to be handled before the call
                ifo_vetos = ifo_vetos.union().active if ifo_vetos else []

            vetos[ifo] = np.array(ifo_vetos)

        return vetos
