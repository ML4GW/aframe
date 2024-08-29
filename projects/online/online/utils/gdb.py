import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import bilby
import matplotlib.pyplot as plt
from gwpy.time import tconvert
from ligo.gracedb.rest import GraceDb as _GraceDb
from online.utils.searcher import Event

GdbServer = Literal["local", "playground", "test", "production"]


class GraceDb(_GraceDb):
    """
    GraceDb Client with some extra methods for
    submitting Aframe and Ampfli events

    Args:
        write_dir:
            Local directory where event data will be written
            upon submission
    """

    def __init__(self, *args, write_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_dir = write_dir

    def submit(self, event: Event):
        logging.info(f"Submitting trigger to file {event.filename}")
        event_dir = self.write_dir / f"event_{int(event.gpstime)}"
        filename = event_dir / event.filename
        response = self.create_event(
            group="CBC",
            pipeline="aframe",
            filename=str(filename),
            search="AllSky",
        )

        # record latencies for this event
        submission_time = float(tconvert(datetime.now(tz=timezone.utc)))
        t_write = event.get_frame_write_time()

        # time to submit since event occured and since the file was written
        total_latency = submission_time - event.gpstime
        write_latency = t_write - event.gpstime
        aframe_latency = submission_time - t_write

        latency_fname = event_dir / "latency.log"
        latency = "Total Latency (s),Write Latency (s),Aframe Latency (s)\n"
        latency += f"{total_latency},{write_latency},{aframe_latency}"
        with open(latency_fname, "w") as f:
            f.write(latency)

        return response

    def submit_pe(
        self,
        result: bilby.core.result.Result,
        mollview_plot: plt.figure,
        graceid: int,
    ):
        corner_fname = self.write_dir / "corner_plot.png"
        result.plot_corner(filename=corner_fname)
        self.write_log(
            graceid, "Corner plot", filename=corner_fname, tag_name="pe"
        )

        mollview_fname = self.write_dir / "mollview_plot.png"
        mollview_plot.savefig(mollview_fname, dpi=300)
        self.write_log(
            graceid,
            "Mollview projection",
            filename=mollview_fname,
            tag_name="sky_loc",
        )

    def submit_pastro(self, pastro: float, graceid: int):
        fname = self.write_dir / "aframe.pastro.json"
        pastro = {
            "BBH": pastro,
            "Terrestrial": 1 - pastro,
            "NSBH": 0,
            "BNS": 0,
        }

        with open(fname, "w") as f:
            json.dump(pastro, f)

        self.write_log(
            graceid,
            "Aframe p_astro",
            filename=fname,
            tag_name="p_astro",
        )


class LocalGraceDb(GraceDb):
    """
    Mock GraceDB client that just writes events locally
    """

    def create_event(self, filename: str, **_):
        return filename

    def write_log(self, filename: str, **_):
        return filename


def gracedb_factory(server: GdbServer, write_dir: Path) -> GraceDb:
    if server == "local":
        return LocalGraceDb(write_dir=write_dir)

    if server in ["playground", "test"]:
        server = f"https://gracedb-{server}.ligo.org/api/"
    elif server == "production":
        server = "https://gracedb.ligo.org/api/"
    else:
        raise ValueError(f"Unknown GraceDB server: {server}")
    return GraceDb(service_url=server, write_dir=write_dir)
