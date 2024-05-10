import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import numpy as np
from gwpy.time import tconvert
from ligo.gracedb.rest import GraceDb
from online_deployment.dataloading import get_prefix

from aframe.analysis.ledger.events import EventSet

Gdb = Literal["local", "playground", "test", "production"]
SECONDS_PER_YEAR = 31556952  # 60 * 60 * 24 * 365.2425


def gps_from_timestamp(timestamp: float):
    return float(tconvert(datetime.fromtimestamp(timestamp, tz=timezone.utc)))


def get_frame_write_time(
    gpstime: float,
    datadir: Path,
    ifos: List[str],
    ifo_suffix: str = None,
):
    t_write = 0
    if ifo_suffix is not None:
        ifo_dir = "_".join([ifos[0], ifo_suffix])
    else:
        ifo_dir = ifos[0]
    prefix, length, _ = get_prefix(datadir / ifo_dir)
    middle = "_".join(prefix.split("_")[1:])
    for ifo in ifos:
        if ifo_suffix is not None:
            ifo_dir = "_".join([ifo, ifo_suffix])
        else:
            ifo_dir = ifo
        prefix = f"{ifo[0]}-{ifo}_{middle}"
        fname = datadir / ifo_dir / f"{prefix}-{int(gpstime)}-{length}.gwf"
        t_write = max(t_write, os.path.getmtime(fname))

    return gps_from_timestamp(t_write)


@dataclass
class Event:
    gpstime: float
    detection_statistic: float
    far: float

    def __str__(self):
        return (
            "Event("
            f"gpstime={self.gpstime:0.3f}, "
            f"detection_statistic={self.detection_statistic:0.2f}, "
            f"far={self.far:0.3e} Hz"
            ")"
        )


class Searcher:
    def __init__(
        self,
        outdir: Path,
        fars_per_day: List[float],
        inference_sampling_rate: float,
        refractory_period: float,
    ) -> None:
        logging.debug("Loading background measurements")
        background_file = outdir / "infer" / "background.h5"
        self.background = EventSet.read(background_file)
        self.min_far = 1 / self.background.Tb * SECONDS_PER_YEAR

        fars_per_day = np.sort(fars_per_day)
        num_events = np.floor(fars_per_day * self.background.Tb / 3600 / 24)
        num_events = num_events.astype(int)
        idx = np.where(num_events == 0)[0]
        if idx:
            raise ValueError(
                "Background livetime {}s not enough to detect "
                "events with daily false alarm rate of {}".format(
                    self.background.Tb, ", ".join(fars_per_day[idx])
                )
            )

        events = np.sort(self.background.detection_statistic)
        self.thresholds = events[-num_events]
        for threshold, far in zip(self.thresholds, fars_per_day):
            logging.info(
                "FAR {}/day threshold is {:0.3f}".format(far, threshold)
            )

        self.inference_sampling_rate = inference_sampling_rate
        self.refractory_period = refractory_period
        self.last_detection_time = time.time() - self.refractory_period
        self.detecting = False

    def check_refractory(self, value):
        time_since_last = time.time() - self.last_detection_time
        if time_since_last < self.refractory_period:
            logging.warning(
                "Detected event with detection statistic {:0.3f} "
                "but it has only been {:0.2f}s since last detection, "
                "so skipping".format(value, time_since_last)
            )
            return True
        return False

    def build_event(self, value: float, t0: float, idx: int):
        if self.check_refractory(value):
            return None

        timestamp = t0 + idx / self.inference_sampling_rate
        far = max(self.background.far(value), self.min_far)
        far /= SECONDS_PER_YEAR

        logging.info(
            "Event coalescence time found to be {:0.3f} "
            "with FAR {:0.3e} Hz".format(timestamp, far)
        )

        self.last_detection_time = time.time()
        return Event(timestamp, value, far)

    def search(self, y: np.ndarray, t0: float):
        """
        Search for above-threshold events in the
        timeseries of integrated network outputs
        `y`. `t0` should represent the timestamp
        of the last sample of *input* to the
        *neural network* that represents the
        *first sample* of the integration window.
        """

        # if we're already mid-detection, take as
        # the event the max in the current window
        max_val = y.max()
        if self.detecting:
            idx = np.argmax(y)
            self.detecting = False
            return self.build_event(max_val, t0, idx)

        # otherwise check all of our thresholds to
        # see if we have an event relative to any of them
        if not (max_val >= self.thresholds).any():
            # if not, nothing to do here
            return None

        logging.info(
            f"Detected event with detection statistic>={max_val:0.3f}"
        )

        # check if the integrated output is still
        # ramping as we get to the end of the frame
        idx = np.argmax(y)
        if idx < (len(y) - 1):
            # if not, assume the event is in this
            # frame and build an event around it
            return self.build_event(max_val, t0, idx)
        else:
            # otherwise, note that we're mid-event but
            # wait until the next frame to mark it
            self.detecting = True
            return None


@dataclass
class LocalGdb:
    def createEvent(self, filename: str, **_):
        return filename

    def writeLog(self, filename: str, **_):
        return filename


class Trigger:
    def __init__(self, server: Gdb, write_dir: Path) -> None:
        self.write_dir = write_dir

        if server in ["playground", "test"]:
            server = f"https://gracedb-{server}.ligo.org/api/"
        elif server == "production":
            server = "https://gracedb.ligo.org/api/"
        elif server == "local":
            self.gdb = LocalGdb()
            return
        else:
            raise ValueError(f"Unknown server {server}")
        self.gdb = GraceDb(service_url=server)

    def __post_init__(self):
        self.write_dir.mkdir(exist_ok=True, parents=True)

    def submit(
        self,
        event: Event,
        ifos: List[str],
        datadir: Path,
        ifo_suffix: str = None,
    ):
        gpstime = event.gpstime
        event_dir = self.write_dir / f"event_{int(gpstime)}"
        event_dir.mkdir(exist_ok=True, parents=True)
        filename = event_dir / f"event-{int(gpstime)}.json"

        event = asdict(event)
        event["ifos"] = ifos
        filecontents = str(event)
        filecontents = json.loads(filecontents.replace("'", '"'))
        with open(filename, "w") as f:
            json.dump(filecontents, f)

        logging.info(f"Submitting trigger to file {filename}")
        response = self.gdb.createEvent(
            group="CBC",
            pipeline="aframe",
            filename=str(filename),
            search="AllSky",
        )
        submission_time = float(tconvert(datetime.now(tz=timezone.utc)))
        t_write = get_frame_write_time(gpstime, datadir, ifos, ifo_suffix)
        # Time to submit since event occured and since the file was written
        total_latency = submission_time - gpstime
        write_latency = t_write - gpstime
        aframe_latency = submission_time - t_write

        latency_fname = event_dir / "latency.log"
        latency = "Total Latency (s),Write Latency (s),Aframe Latency (s)\n"
        latency += f"{total_latency},{write_latency},{aframe_latency}"
        with open(latency_fname, "w") as f:
            f.write(latency)

        return response

    def submit_pe(self, bilby_result, mollview_plot, graceid):
        corner_fname = self.write_dir / "corner_plot.png"
        bilby_result.plot_corner(filename=corner_fname)
        self.gdb.writeLog(
            graceid, "Corner plot", filename=corner_fname, tag_name="pe"
        )

        mollview_fname = self.write_dir / "mollview_plot.png"
        mollview_plot.savefig(mollview_fname, dpi=300)
        self.gdb.writeLog(
            graceid,
            "Mollview projection",
            filename=mollview_fname,
            tag_name="sky_loc",
        )

        self.gdb.writeLog(graceid, "O3 Replay")
