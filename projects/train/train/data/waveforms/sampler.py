from pathlib import Path

import h5py
import numpy as np
import torch
from ledger.injections import WaveformSet, waveform_class_factory
from utils import x_per_y

Distribution = torch.distributions.Distribution


def waveform_files(path: Path) -> list[Path]:
    """The file at `path`, or every file in the directory `path`, e.g. one
    per population, in name order.

    If `path` doesn't exist but the same path ending in `.hdf5` does, that
    file is used, so a config written for per-population directories
    (e.g. val_waveforms/) still reads an older single file
    (val_waveforms.hdf5).
    """
    path = Path(path)
    if not path.exists() and path.with_suffix(".hdf5").exists():
        path = path.with_suffix(".hdf5")
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            raise ValueError(f"{path} holds no waveform files")
        return files
    return [path]


def check_waveform_files(fnames: list[Path], cls) -> tuple[float, int]:
    """The right_pad of waveform files, and the length in samples of their
    shortest waveforms.

    Files can hold waveforms of different lengths, e.g. one population
    each. Training only uses the end of a waveform, where the merger is
    right_pad seconds from the end, so waveforms are all cut to the shortest
    by keeping their ends, which needs every file's right_pad to match.
    """
    right_pads, sizes = set(), []
    for fname in fnames:
        # only the first waveform, plus the file's attributes
        first = cls.read_idx(fname, np.arange(1))
        right_pads.add(float(first.right_pad))
        sizes.append(round(first.duration * first.sample_rate))
    if len(right_pads) != 1:
        raise ValueError(f"{fnames} have different right_pads {right_pads}")
    return right_pads.pop(), min(sizes)


class WaveformSampler(torch.nn.Module):
    """
    Base object defining methods that waveform producing classes
    should implement. Should not be instantiated on its own.
    Args:
        ifos:
            List of interferometers that are being trained on.
        sample_rate:
            Sample rate in Hz of generated waveforms
        val_waveform_file:
            Path to the validation waveforms file, or to a directory
            of them, e.g. one per population. See `check_waveform_files`.
    """

    def __init__(
        self,
        *args,
        ifos: list[str],
        sample_rate: float,
        val_waveform_file: Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ifos = ifos
        self.sample_rate = sample_rate
        self.val_waveform_file = waveform_files(val_waveform_file)
        self.right_pad, self.val_waveform_size = check_waveform_files(
            self.val_waveform_file, self.waveform_set_cls
        )

    @property
    def waveform_set_cls(self):
        cls = waveform_class_factory(
            self.ifos,
            WaveformSet,
            "WaveformSet",
        )
        return cls

    def get_slice_bounds(self, total, world_size, rank) -> tuple[int, int]:
        """
        Determine waveform indices to load for this device
        given our rank and world size
        """
        per_dev = x_per_y(abs(total), world_size)
        start = rank * per_dev
        stop = (rank + 1) * per_dev
        return start, stop

    # Assuming that we're going to be loading validation waveforms
    # from disk for now, so this function can be defined here.
    def get_val_waveforms(self, world_size, rank):
        """
        Returns validation waveforms for this device: its share of each
        file's, cut to val_waveform_size samples by keeping their ends.
        Only those samples are read, so long waveforms aren't loaded whole.
        """
        size, waveforms = self.val_waveform_size, []
        for fname in self.val_waveform_file:
            with h5py.File(fname, "r") as f:
                start, stop = self.get_slice_bounds(
                    f.attrs["length"], world_size, rank
                )
                waveform_set = self.waveform_set_cls.read_idx(
                    fname, np.arange(start, min(stop, f.attrs["length"]))
                )
                waveforms.append(waveform_set.waveforms[..., -size:])
        return torch.Tensor(np.concatenate(waveforms))

    def get_test_waveforms(self):
        raise NotImplementedError

    def sample(self):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
