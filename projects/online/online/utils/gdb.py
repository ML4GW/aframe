import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import bilby
import healpy as hp
import matplotlib.pyplot as plt
from gwpy.time import tconvert
from ligo.gracedb.rest import GraceDb as _GraceDb
from ligo.skymap.tool import ligo_skymap_from_samples

from online.utils.searcher import Event

if TYPE_CHECKING:
    from astropy.io.fits import BinTableHDU

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
        super().__init__(*args, **kwargs, use_auth="scitoken")
        self.write_dir = write_dir

    def submit(self, event: Event):
        logging.info(f"Submitting trigger to file {event.filename}")
        event_dir = self.write_dir / f"event_{int(event.gpstime)}"
        filename = event_dir / event.filename
        logging.info("Creating event in GraceDB")
        response = self.create_event(
            group="CBC",
            pipeline="aframe",
            filename=str(filename),
            search="AllSky",
        )
        logging.info("Event created")

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

    def submit_low_latency_pe(
        self,
        result: bilby.core.result.Result,
        skymap: "BinTableHDU",
        graceid: int,
        event_time: float,
    ):
        event_dir = self.write_dir / f"event_{int(event_time)}"
        skymap_fname = event_dir / "amplfi.fits"
        skymap.writeto(skymap_fname)
        logging.info("Submitting skymap to GraceDB")
        self.write_log(graceid, "skymap", filename=skymap_fname, tag_name="pe")
        logging.info("Skymap submitted")

        posterior_fname = event_dir / "posterior_samples.dat"
        result.save_posterior_samples(posterior_fname)
        corner_fname = event_dir / "corner_plot.png"
        result.plot_corner(
            parameters=["chirp_mass", "mass_ratio", "distance"],
            filename=corner_fname,
        )
        logging.info("Submitting corner plot to GraceDB")
        self.write_log(
            graceid, "Corner plot", filename=corner_fname, tag_name="pe"
        )
        logging.info("Corner plot submitted")

        mollview_fname = event_dir / "mollview_plot.png"
        fig = plt.figure()
        title = (f"{event_time:.3} sky map",)
        hp.mollview(
            skymap.data["PROBDENSITY"], fig=fig, title=title, hold=True
        )
        plt.close()
        fig.savefig(mollview_fname, dpi=300)
        logging.info("Submitting Mollview plot to GraceDB")
        self.write_log(
            graceid,
            "Mollview projection",
            filename=mollview_fname,
            tag_name="sky_loc",
        )
        logging.info("Mollview plot submitted")

    def submit_ligo_skymap_from_samples(
        self,
        result: bilby.core.result.Result,
        graceid: int,
        event_time: float,
    ):
        event_dir = self.write_dir / f"event_{int(event_time)}"
        filename = event_dir / "posterior_samples.dat"
        result.save_posterior_samples(filename=filename)

        # TODO: we probably will need cores beyond whats on the head node
        # to really speed this up, and also the following env variables
        # need to be set to take advantage of thread parallelism:
        # {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}.
        # we might need to submit this via condor
        args = [str(filename), "-j", str(8), "-o", str(event_dir)]

        ligo_skymap_from_samples.main(args)
        self.write_log(
            graceid,
            "ligo-skymap-from-samples",
            filename=str(event_dir / "skymap.fits"),
            tag_name="sky_loc",
        )
        logging.info("Ligo-skymap-from-samples skymap submitted")

    def submit_pastro(self, pastro: float, graceid: int, event_time: float):
        event_dir = self.write_dir / f"event_{int(event_time)}"
        fname = event_dir / "aframe.pastro.json"
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

    def write_log(self, *args, **kwargs):
        pass


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


def authenticate():
    # TODO: don't hardcode keytab locations
    subprocess.run(
        [
            "kinit",
            "aframe-1-scitoken/robot/ldas-pcdev12.ligo.caltech.edu@LIGO.ORG",
            "-k",
            "-t",
            os.path.expanduser(
                "~/robot/aframe-1-scitoken_robot_ldas-pcdev12.ligo.caltech.edu.keytab"  # noqa
            ),
        ]
    )
    subprocess.run(
        [
            "htgettoken",
            "-v",
            "-a",
            "vault.ligo.org",
            "-i",
            "igwn",
            "-r",
            "aframe-1-scitoken",
            "--scopes=gracedb.read",
            "--credkey=aframe-1-scitoken/robot/ldas-pcdev12.ligo.caltech.edu",
            "--nooidc",
        ]
    )
