import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from gwpy.time import tconvert

from ledger.events import EventSet
from online.utils.dataloading import get_prefix

SECONDS_PER_YEAR = 31556952  # 60 * 60 * 24 * 365.2425


def gps_from_timestamp(timestamp: float):
    return float(tconvert(datetime.fromtimestamp(timestamp, tz=timezone.utc)))


@dataclass
class Event:
    gpstime: float
    detection_statistic: float
    far: float
    ifos: List[str]
    channel: str
    datadir: Path
    ifo_suffix: Optional[str] = None

    def get_frame_write_time(self):
        """
        Get the time the frame corresponding
        to this event was written to disk
        """
        t_write = 0
        if self.ifo_suffix is not None:
            ifo_dir = "_".join([self.ifos[0], self.ifo_suffix])
        else:
            ifo_dir = self.ifos[0]
        prefix, length, _ = get_prefix(self.datadir / ifo_dir)
        middle = "_".join(prefix.split("_")[1:])
        for ifo in self.ifos:
            if self.ifo_suffix is not None:
                ifo_dir = "_".join([ifo, self.ifo_suffix])
            else:
                ifo_dir = ifo
            prefix = f"{ifo[0]}-{ifo}_{middle}"
            fname = (
                self.datadir
                / ifo_dir
                / f"{prefix}-{int(self.gpstime)}-{length}.gwf"
            )
            t_write = max(t_write, os.path.getmtime(fname))

        return gps_from_timestamp(t_write)

    def __str__(self):
        return (
            "Event("
            f"gpstime={self.gpstime:0.3f}, "
            f"detection_statistic={self.detection_statistic:0.2f}, "
            f"far={self.far:0.3e} Hz, "
            f"ifos={self.ifos}, "
            f"channel={self.channel}"
            ")"
        )

    @property
    def filename(self):
        return f"event_{int(self.gpstime)}.json"

    def write(self, directory: Path):
        """
        Write event information to a local directory
        """
        path = directory / f"event_{int(self.gpstime)}"
        path.mkdir(exist_ok=True, parents=True)
        filename = path / self.filename

        event = asdict(self)
        _, _ = event.pop("datadir"), event.pop("ifo_suffix", None)
        filecontents = str(event)
        filecontents = json.loads(filecontents.replace("'", '"'))
        with open(filename, "w") as f:
            json.dump(filecontents, f)

        return path


class Searcher:
    """
    Object for managing aframe search state, building aframe events,
    """

    def __init__(
        self,
        background: EventSet,
        far_threshold: float,
        inference_sampling_rate: float,
        refractory_period: float,
        ifos: List[str],
        channel: str,
        datadir: Path,
        ifo_suffix: Optional[str] = None,
    ) -> None:
        self.inference_sampling_rate = inference_sampling_rate
        self.refractory_period = refractory_period
        self.ifos = ifos
        self.channel = channel
        self.datadir = datadir
        self.ifo_suffix = ifo_suffix

        # initialize the state of the searcher:

        # flag that declares if we're in the process
        # of detecting an event between frames
        self.detecting = False

        self.last_detection_time = time.time() - self.refractory_period

        # calculate the detection statistic threshold
        # corresponding to the requested FAR threshodl
        self.background = background
        self.threshold = self.background.threshold_at_far(far_threshold)

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
        far = self.background.far(value)
        far /= SECONDS_PER_YEAR

        logging.info(
            "Event coalescence time found to be {:0.3f} "
            "with FAR {:0.3e} Hz".format(timestamp, far)
        )

        self.last_detection_time = time.time()
        return Event(
            gpstime=timestamp,
            detection_statistic=value,
            far=far,
            ifos=self.ifos,
            channel=self.channel,
            datadir=self.datadir,
            ifo_suffix=self.ifo_suffix,
        )

    def search(self, y: np.ndarray, t0: float) -> Optional[Event]:
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

        # otherwise, check if the event is above threshold
        if not max_val >= self.threshold:
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
