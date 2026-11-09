import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ledger.events import EventSet

SECONDS_PER_YEAR = 31556952  # 60 * 60 * 24 * 365.2425


@dataclass
class Event:
    gpstime: float
    detection_statistic: float
    far: float
    ifos: List[str]
    channels: List[str]

    def __str__(self):
        return (
            "Event("
            f"gpstime={self.gpstime:0.3f}, "
            f"detection_statistic={self.detection_statistic:0.2f}, "
            f"far={self.far:0.3e} Hz, "
            f"ifos={self.ifos}, "
            f"channels={self.channels}"
            ")"
        )

    @property
    def filename(self):
        return f"event_{int(self.gpstime)}.json"

    @property
    def event_dir(self):
        return Path(f"event_{int(self.gpstime)}")

    def write(self, directory: Path):
        """
        Write event information to a local directory
        """
        path = directory / f"event_{int(self.gpstime)}"
        path.mkdir(exist_ok=True, parents=True)
        filename = path / self.filename

        event = asdict(self)
        event["ifos"] = ",".join(event["ifos"])
        event["end_time"] = int(self.gpstime)
        event["end_time_ns"] = int(self.gpstime % 1 * 1e9)
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
        online_inference_rate: float,
        refractory_period: float,
        ifos: List[str],
        channels: str,
    ) -> None:
        self.online_inference_rate = online_inference_rate
        self.refractory_period = refractory_period
        # Take only the first two ifos/channels for H1/L1
        # Hard-coding this until there's an HLV Aframe model
        self.ifos = ifos[:2]
        self.channels = channels[:2]

        # initialize the state of the searcher:

        # flag that declares if we're in the process
        # of detecting an event between blocks
        self.detecting = False

        self.last_detection_time = 0

        # calculate the detection statistic threshold
        # corresponding to the requested FAR threshold
        self.threshold = background.threshold_at_far(far_threshold)
        # Speed up FAR calculation by excluding below-threshold events
        mask = background.detection_statistic >= self.threshold
        self.background = background[mask]

    def check_refractory(self, timestamp, value):
        time_since_last = timestamp - self.last_detection_time
        if time_since_last < self.refractory_period:
            logging.warning(
                "Detected event with detection statistic {:0.3f} "
                "but it has only been {:0.2f}s since last detection, "
                "so skipping".format(value, time_since_last)
            )
            return True
        return False

    def build_event(self, value: float, t0: float, idx: int):
        timestamp = t0 + idx / self.online_inference_rate

        if self.check_refractory(timestamp, value):
            return None

        logging.debug("Computing FAR")
        far = self.background.far(value)
        far /= SECONDS_PER_YEAR
        logging.debug("FAR computed")

        logging.info(
            "Event coalescence time found to be {:0.3f} "
            "with FAR {:0.3e} Hz".format(timestamp, far)
        )

        self.last_detection_time = timestamp
        event = Event(
            gpstime=timestamp,
            detection_statistic=value,
            far=far,
            ifos=self.ifos,
            channels=self.channels,
        )
        # reset state to not detecting
        # after we've detected an event
        self.detecting = False
        return event

    def search(
        self,
        significance_outputs: np.ndarray,
        timing_outputs: np.ndarray,
        t0: float,
    ) -> Optional[Event]:
        """
        Search for above-threshold events in the
        timeseries of integrated network outputs
        `significance_outputs` and use the peak
        index of `timing_outputs` to estimate a
        merger time. `t0` should represent the
        timestamp of the last sample of *input*
        to the *neural network* that represents the
        *first sample* of the integration window.
        """

        max_val = significance_outputs.max()
        # check if the event is above threshold
        if not max_val >= self.threshold:
            # if not, nothing to do here
            return None

        # check if the integrated output is still
        # ramping as we get to the end of the block
        idx = np.argmax(timing_outputs)
        if idx < (len(timing_outputs) - 1):
            # if not, assume the event is in this
            # block and build an event around it
            logging.info(
                f"Detected event with detection statistic {max_val:0.3f}"
            )
            return self.build_event(max_val, t0, idx)
        else:
            # otherwise, note that we're mid-event but
            # wait until the next block to mark it
            logging.info(
                f"Event with detection statistic {max_val:0.3f} "
                "found but still ramping, waiting for next block"
            )
            self.detecting = True
            return None
