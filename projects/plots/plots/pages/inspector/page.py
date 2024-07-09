from plots.data import Data
from typing import TYPE_CHECKING, List, Sequence
import torch
from pathlib import Path
from .utils import get_strain_fname 
import numpy as np

if TYPE_CHECKING:
    from utils.preprocessing import BatchWhitener, BackgroundSnapshotter


class EventAnalyzer:
    def __init__(
        self, 
        model: torch.nn.Module,
        whitener: BatchWhitener,
        snapshotter: BackgroundSnapshotter,
        data_dir: Path,

    ):
        self.model = model
        self.whitener = whitener
        self.snapshotter = snapshotter
        self.data_dir = data_dir


    def find_strain(self, time: float, shifts: Sequence[float]):
        fname, t0, duration = get_strain_fname(self.data_dir, time)
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)

    def analyze(
        self, 
        time: float, 
        shifts: List[float], 
        foreground: bool
    ):
        pass


class InspectorPlot:
    def __init__(self, data: Data):
        self.data = Data

    
