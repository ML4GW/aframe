import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from vizapp import path_utils

from bbhnet.analysis.distributions import ClusterDistribution

event_type = "(foreground|background)"
optional_pattern = r"_norm-seconds=[0-9]{1,4}(\.[0-9])?"
dist_pattern = re.compile(rf"{event_type}(?=({optional_pattern})?\.h5)")
norm_pattern = re.compile(r"(?<=_norm-seconds=)[0-9]{1,4}(\.[0-9])?")


@dataclass
class AnalysisResults:
    foreground: ClusterDistribution
    background: ClusterDistribution


def load_results(data_dir: Path) -> Dict[Optional[float], AnalysisResults]:
    distributions = defaultdict(dict)
    for fname in data_dir.iterdir():
        match = dist_pattern.search(fname.name)
        if match is None:
            continue

        distribution = ClusterDistribution.from_file(
            "integrated", ["H1", "L1"], fname
        )

        event_type = match.group(0)
        match = norm_pattern.search(fname.name)
        norm = float(match.group(0)) if match is not None else None
        distributions[norm][event_type] = distribution

    return {k: AnalysisResults(**v) for k, v in distributions.items()}


@dataclass
class Foreground:
    detection_statistics: np.ndarray
    event_times: np.ndarray
    shifts: np.ndarray
    fars: np.ndarray
    snrs: np.ndarray
    distances: np.ndarray
    m1s: np.ndarray
    m2s: np.ndarray
    time_deltas: np.ndarray

    @property
    def chirps(self) -> np.ndarray:
        return (self.m1s * self.m2s) ** 0.6 / (self.m1s + self.m2s) ** 0.2


def get_foreground(
    results: AnalysisResults, data_dir: Path, norm: Optional[float] = None
) -> Foreground:
    int_dirname = path_utils.get_response_dirname("foreground", norm)

    events = defaultdict(list)
    for shift in data_dir.iterdir():
        if not (shift.is_dir() and shift.name.startswith("dt")):
            continue

        shift_vals = path_utils.dirname_to_shifts(shift.name)
        mask = (results.foreground.shifts == shift_vals).all(axis=1)

        with h5py.File(shift / "injection" / "params.h5") as f:
            event_times = f["geocent_time"][:]
            h1_snrs = f["H1_snr"][:]
            l1_snrs = f["L1_snr"][:]
            m1s = f["mass_1"][:]
            m2s = f["mass_2"][:]
            distances = f["luminosity_distance"][:]
            network_snrs = (h1_snrs**2 + l1_snrs**2) ** 0.5

        if norm is not None:
            segments = path_utils.get_segments(shift / int_dirname)

        # for each injection, find the event in the
        # foreground which is closest to it
        for i, t in enumerate(event_times):
            if norm is not None:
                # if we have a normalization window, ignore injections
                # that fall within the initial normalization window
                for start, stop in segments:
                    if start < t < stop:
                        break
                else:
                    continue

                if t < (start + norm):
                    continue

            # only look at events from the current shift
            fgd_times = results.foreground.event_times[mask]

            # find the event in the foreground that's closest
            # to the indicated event time, accounting for the 1s
            # discrepancy we expect from accumulation
            # TODO: generalize to kernel_length
            diff = np.abs(fgd_times - t - 1)
            idx = np.argmin(diff)

            # now grab all the data from the foreground event
            # at the corresponding index
            diff = diff[idx]
            event = results.foreground.events[mask][idx]
            far = results.background.far(event)
            event_time = fgd_times[idx]

            # now grab some injection metadata from this event
            m1 = m1s[i]
            m2 = m2s[i]
            snr = network_snrs[i]
            distance = distances[i]

            events["detection_statistics"].append(event)
            events["time_deltas"].append(diff)
            events["fars"].append(far)
            events["event_times"].append(event_time)
            events["shifts"].append(shift.name)
            events["snrs"].append(snr)
            events["m1s"].append(m1)
            events["m2s"].append(m2)
            events["distances"].append(distance)

    events = {k: np.array(v) for k, v in events.items()}
    return Foreground(**events)
