from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import h5py
import numpy as np
import torch
from ledger.injections import waveform_class_factory
from plots.data import Data

from .utils import get_strain_fname

if TYPE_CHECKING:
    from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


class EventAnalyzer:
    def __init__(
        self,
        model: torch.nn.Module,
        whitener: BatchWhitener,
        snapshotter: BackgroundSnapshotter,
        data_dir: Path,
        ifos: List[str],
    ):
        self.model = model
        self.whitener = whitener
        self.snapshotter = snapshotter
        self.data_dir = data_dir

    @property
    def waveform_class(self):
        return waveform_class_factory(self.ifos)

    def find_strain(self, time: float, shifts: Sequence[float]):
        # find strain file corresponding to requested time
        fname, t0, duration = get_strain_fname(self.data_dir, time)
        start, stop = None
        # find indices of data needed for inference
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)
        # start, stop = get_indices(
        #    times,
        #    time - self.length_previous_data,
        #    time + self.padding + (self.fduration / 2),
        # )
        # times = times[start:stop]
        strain = []
        with h5py.File(fname, "r") as f:
            for ifo, shift in zip(self.ifos, shifts):
                shift_size = int(shift * self.sample_rate)
                start_shifted, stop_shifted = (
                    start + shift_size,
                    stop + shift_size,
                )
                data = torch.tensor(f[ifo][start_shifted:stop_shifted])
                strain.append(data)

        return torch.stack(strain, axis=0), times

    def find_waveform(self, time: float, shifts: np.ndarray):
        """
        find the closest injection that corresponds to event
        time and shifts from waveform dataset
        """
        waveform = self.waveform_class.read(
            self.response_path, time - 0.1, time + 0.1, shifts
        )
        return waveform

    def inject(self):
        pass

    def analyze(self, time: float, shifts: List[float], foreground: bool):
        pass


class InspectorPlot:
    def __init__(self, data: Data):
        self.data = Data
