import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
import bilby
import h5py
from gwpy.time import tconvert
from ligo.gracedb.rest import GraceDb as _GraceDb
from ..subprocesses.utils import run_subprocess_with_logging
from ligo.skymap.tool.ligo_skymap_plot import main as ligo_skymap_plot
from online.utils.searcher import Event
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from astropy.io.fits import BinTableHDU

from enum import Enum


class GdbServer(Enum):
    """Enum for GraceDB servers."""

    local = None
    playground = "https://gracedb-playground.ligo.org/"
    test = "https://gracedb-test.ligo.org/"
    production = "https://gracedb.ligo.org/"
    test01 = "https://gracedb-test01.igwn.org/"

    @property
    def service_url(self) -> str:
        if self == GdbServer.local:
            return self.name
        return self.value + "api/"

    def gevent_url(self, graceid: str) -> str:
        """Get the URL for a specific event on this server."""
        if self == GdbServer.local:
            return graceid
        return self.value + f"events/{graceid}/view"

    def create_gracedb(self, write_dir: Path, **kwargs) -> "GraceDb":
        """Create a GraceDb client instance for this server."""
        if self == GdbServer.local:
            return LocalGraceDb(server=self, write_dir=write_dir)

        return GraceDb(
            server=self,
            write_dir=write_dir,
            **kwargs,
        )


class GraceDb(_GraceDb):
    """
    GraceDb Client with some extra methods for
    submitting Aframe and Ampfli events

    Args:
        write_dir:
            Local directory where event data will be written
            upon submission
        logger:
            Optional logger object to emit logs
    """

    def __init__(
        self,
        *args,
        server: GdbServer,
        write_dir: Path,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            service_url=server.service_url,
            use_auth="scitoken",
        )
        self.server = server
        self.write_dir = write_dir
        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    def submit(self, event: Event):
        self.logger.info(f"Submitting trigger to file {event.filename}")
        event_dir = self.write_dir / event.event_dir
        filename = event_dir / event.filename
        self.logger.info("Creating event in GraceDB")
        response = self.create_event(
            group="CBC",
            pipeline="aframe",
            filename=str(filename),
            search="AllSky",
        )

        self.logger.debug("Event created")

        # Get the event's graceid for submitting
        # further data products
        if self.server == GdbServer.local:
            # The local gracedb client just returns the filename
            graceid = response
        else:
            graceid = response.json()["graceid"]

        url = self.server.gevent_url(graceid)
        filename = event_dir / "gracedb_url.txt"
        with open(filename, "w") as f:
            f.write(url)

        # record latencies for this event;
        # TODO: determine underlying issue here
        # Handle issue where sometimes the pipeline lags,
        # and the frame file has already left the buffer
        submission_time = float(tconvert(datetime.now(tz=timezone.utc)))
        try:
            t_write = event.get_frame_write_time()
        except FileNotFoundError:
            self.logger.warning(
                "Could not find frame file to evaluate write time. "
                "Using rounded down event gpstime instead."
            )
            t_write = int(event.gpstime)
        except ValueError:
            self.logger.warning(
                "Either too many prefixes or durations in data directory"
                "If running in online mode, be careful"
            )
            t_write = int(event.gpstime)

        # time to submit since event occured and since the file was written
        total_latency = submission_time - event.gpstime
        write_latency = t_write - event.gpstime
        aframe_latency = submission_time - t_write

        latency_fname = event_dir / "latency.log"
        latency = "Total Latency (s),Write Latency (s),Aframe Latency (s)\n"
        latency += f"{total_latency},{write_latency},{aframe_latency}"
        with open(latency_fname, "w") as f:
            f.write(latency)

        return graceid

    def submit_low_latency_pe(
        self,
        result: bilby.core.result.Result,
        skymap: "BinTableHDU",
        graceid: int,
        event_dir: Path,
    ):
        event_dir = self.write_dir / event_dir
        skymap_fname = event_dir / "amplfi.multiorder.fits"
        skymap.writeto(skymap_fname)

        self.logger.debug("Submitting skymap to GraceDB")
        self.write_log(
            graceid,
            "skymap",
            filename=skymap_fname,
            tag_name="sky_loc",
            label="SKYMAP_READY",
        )
        self.logger.debug("Skymap submitted")

        # rename so we can later write kde file
        skymap_fname.rename(event_dir / "amplfi.multiorder.fits,0")

        # Write posterior samples to file, adhering to expected format
        filename = event_dir / "amplfi.posterior_samples.hdf5"
        posterior_df = result.posterior
        columns = list(posterior_df.columns)
        columns.remove("distance")
        posterior_df = posterior_df[columns]
        posterior_samples = posterior_df.to_records(index=False)
        with h5py.File(filename, "w") as f:
            f.create_dataset("posterior_samples", data=posterior_samples)
        self.write_log(graceid, "posterior", filename=filename, tag_name="pe")

        corner_fname = event_dir / "corner_plot.png"
        result.plot_corner(
            parameters=[
                "chirp_mass",
                "mass_ratio",
                "luminosity_distance",
                "mass_1",
                "mass_2",
            ],
            filename=corner_fname,
        )

        self.logger.debug("Submitting corner plot to GraceDB")
        self.write_log(
            graceid, "Corner plot", filename=corner_fname, tag_name="pe"
        )
        self.logger.debug("Corner plot submitted")

    def submit_ligo_skymap_from_samples(
        self,
        result: bilby.core.result.Result,
        graceid: int,
        event_dir: Path,
        ifos: List[str],
    ):
        event_dir = self.write_dir / event_dir
        filename = event_dir / "posterior_samples.dat"
        result.save_posterior_samples(filename=filename)

        # TODO: we probably will need cores beyond whats on the head node
        # to really speed this up, and also the following env variables
        # need to be set to take advantage of thread parallelism:
        # {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}.
        # we might need to submit this via condor
        args = [
            "ligo-skymap-from-samples",
            str(filename),
            "-j",
            str(64),
            "-o",
            str(event_dir),
            "--maxpts",
            str(10000),
            "--fitsoutname",
            "amplfi.multiorder.fits",
            "--instruments",
        ]

        args.extend(ifos)

        # TODO: ligo-skymap-from-samples doesnt clean up
        # process pool on purpose so that overhead from
        # initializing pool can be eliminated. Once
        # we get our own resources we should take
        # advantage of this

        # run subprocess, passing any output to python logger
        result = run_subprocess_with_logging(
            args, logger=self.logger, log_stderr_on_success=True
        )

        self.write_log(
            graceid,
            "ligo-skymap-from-samples",
            filename=str(event_dir / "amplfi.multiorder.fits"),
            tag_name="sky_loc",
            label="SKYMAP_READY",
        )

    def submit_skymap_plots(self, graceid: int, event_dir: Path):
        plt.switch_backend("agg")

        event_dir = self.write_dir / event_dir
        amplfi_fname = str(event_dir / "amplfi.histogram.png")
        ligo_skymap_fname = str(event_dir / "amplfi.kde.png")

        ligo_skymap_plot(
            [
                str(event_dir / "amplfi.multiorder.fits,0"),
                "--annotate",
                "--contour",
                "50",
                "90",
                "-o",
                amplfi_fname,
            ]
        )
        plt.close()
        ligo_skymap_plot(
            [
                str(event_dir / "amplfi.multiorder.fits"),
                "--annotate",
                "--contour",
                "50",
                "90",
                "-o",
                ligo_skymap_fname,
            ]
        )
        plt.close()
        # ligo_skymap_plot_volume(
        #    [
        #        str(event_dir / "amplfi.multiorder.fits"),
        #        "--annotate",
        #        "-o",
        #        str(event_dir / "amplfi.multiorder.volume.png"),
        #    ]
        # )
        # plt.close()
        self.write_log(
            graceid,
            "Molleweide projection of amplfi.multiorder.fits,0",
            filename=amplfi_fname,
            tag_name="sky_loc",
        )

        self.write_log(
            graceid,
            "Molleweide projection of amplfi.multiorder.fits",
            filename=ligo_skymap_fname,
            tag_name="sky_loc",
        )

        # self.write_log(
        #    graceid,
        #    "Volume rendering of amplfi.multiorder.fits",
        #    filename=str(event_dir / "amplfi.multiorder.volume.png"),
        #    tag_name="sky_loc",
        # )

    def submit_pastro(self, pastro: float, graceid: int, event_dir: Path):
        event_dir = self.write_dir / event_dir
        fname = event_dir / "aframe.p_astro.json"
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
            label="PASTRO_READY",
        )


class LocalGraceDb(GraceDb):
    """
    Mock GraceDB client that just writes events locally
    """

    def create_event(self, filename: str, **_):
        return filename

    def write_log(self, *args, **kwargs):
        pass
