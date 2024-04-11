import glob
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional

import h5py
import lightning.pytorch as pl
import ray
import torch
from ledger.injections import LigoWaveformSet

from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import unfold_windows
from train import augmentations as aug
from train.data.utils import fs as fs_utils
from train.metrics import get_timeslides
from utils import x_per_y
from utils.preprocessing import PsdEstimator

Tensor = torch.Tensor


# TODO: using this right now because
# lightning.pytorch.utilities.CombinedLoader
# is not supported when calling `.fit`. Once
# this has been fixed in
# https://github.com/Lightning-AI/lightning/issues/16830,
# we should switch to using a CombinedLoader for validation
class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets, minimum: Optional[int] = None):
        super().__init__()
        self.datasets = datasets
        self.minimum = minimum

    def __len__(self):
        lengths = []
        for dset in self.datasets:
            try:
                lengths.append(len(dset))
            except Exception as e:
                raise e from None
        return self.minimum or min(lengths)

    def __iter__(self):
        return zip(*self.datasets)


class BaseAframeDataset(pl.LightningDataModule):
    def __init__(
        self,
        # data loading args
        data_dir: str,
        ifos: Sequence[str],
        valid_frac: float,
        # preprocessing args
        batch_size: int,
        kernel_length: float,
        fduration: float,
        psd_length: float,
        # augmentation args
        waveform_prob: float = 1,
        snr_thresh: float = 4,
        max_snr: float = 100,
        snr_alpha: float = 3,
        trigger_pad: float = 0,
        fftlength: Optional[float] = None,
        highpass: Optional[float] = None,
        snr_sampler: Optional[Callable[[int], torch.Tensor]] = None,
        # validation args
        valid_stride: Optional[float] = None,
        num_valid_views: int = 4,
        min_valid_duration: float = 15000,
        valid_livetime: float = (3600 * 12),
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_ifos = len(ifos)

        # Set up some of our data augmentation modules
        self.inverter = aug.SignalInverter(0.5)
        self.reverser = aug.SignalReverser(0.5)

        # these are modules that require our data to be
        # downloaded first, either for loading signals
        # or to infer sample rate, so wait to construct
        # them until self.setup
        self.waveform_sampler = None
        self.whitener = None
        self.projector = None
        self.psd_estimator = None
        self._on_device = False
        self.snr_sampler = snr_sampler

        # generate our local node data directory
        # if our specified data source is remote
        self.data_dir = fs_utils.get_data_dir(self.hparams.data_dir)
        self.verbose = verbose

    # ================================================ #
    # Distribution utilities
    # ================================================ #
    def get_local_device(self):
        """
        Get the device string for the device that
        we're actually training on so that we can move
        our transforms on to it because Lightning won't
        do this for us in a DataModule. NOTE!!! This
        method is currently incorrect during ray distributed
        training, but keeping it for posterity in the hopes
        that we can eventually get rid of self._move_to_device
        """
        if not self.trainer.device_ids:
            return "cpu"
        elif len(self.trainer.device_ids) == 1:
            return f"cuda:{self.trainer.device_ids[0]}"
        else:
            _, rank = self.get_world_size_and_rank()
            if ray.is_initialized():
                device_ids = ray.train.torch.get_device()
                if isinstance(device_ids, list):
                    return device_ids[rank]
                return device_ids
            else:
                device_id = self.trainer.device_ids[rank]
                return f"cuda:{device_id}"

    def get_world_size_and_rank(self) -> tuple[int, int]:
        """
        Name says it all, but generalizes to the case
        where we aren't running distributed training.
        """
        if not torch.distributed.is_initialized():
            return 1, 0
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return world_size, rank

    def get_logger(self, world_size, rank):
        logger_name = "AframeDataset"
        if world_size > 1:
            logger_name += f"{rank}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        return logger

    # ================================================ #
    # Re-paramterizing some attributes
    # ================================================ #
    @property
    def sample_length(self) -> float:
        """Length of samples generated by datasets in seconds"""
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

    @property
    def pad_size(self) -> int:
        """
        Number of samples away from edge of kernel to ensure
        that waveforms are injected at.
        """
        return int(self.hparams.trigger_pad * self.sample_rate)

    # TODO: Come up with a more clever scheme for breaking up
    # our training and validation background data
    @property
    def train_fnames(self) -> Sequence[str]:
        fnames = glob.glob(f"{self.data_dir}/background/*.hdf5")
        return sorted(fnames)[:-1]

    @property
    def valid_fnames(self) -> Sequence[str]:
        fnames = glob.glob(f"{self.data_dir}/background/*.hdf5")
        fnames = sorted([Path(fname) for fname in fnames])
        durations = [int(fname.stem.split("-")[-1]) for fname in fnames]
        valid_fnames = []
        valid_duration = 0
        while valid_duration < self.hparams.min_valid_duration:
            fname, duration = fnames.pop(-1), durations.pop(-1)
            valid_duration += duration
            valid_fnames.append(str(fname))
        return valid_fnames

    @property
    def val_batch_size(self):
        """Use larger batch sizes when we don't need gradients."""
        return int(1 * self.hparams.batch_size)

    # ================================================ #
    # Utilities for initial data loading and preparation
    # ================================================ #
    def prepare_data(self):
        """
        Download s3 data if it doesn't exist.
        """
        logger = logging.getLogger("AframeDataset")
        bucket, _ = fs_utils.split_data_dir(self.hparams.data_dir)
        if bucket is None:
            return
        logger.info(
            "Downloading data from S3 bucket {} to {}".format(
                bucket, self.data_dir
            )
        )
        fs_utils.download_training_data(bucket, self.data_dir)

    def load_signals(self, dataset, start, stop):
        """
        Loads waveforms assuming that the coalescence
        is in the middle, but we should really stop
        this. TODO: stop this
        """
        size = int(dataset.shape[-1] // 2)
        pad = int(0.02 * self.sample_rate)
        signals = torch.Tensor(dataset[start:stop, : size + pad])
        self._logger.info("Waveforms loaded")

        cross, plus = signals[:, 0], signals[:, 1]
        return cross, plus

    def get_slice_bounds(self, total, world_size, rank) -> tuple[int, int]:
        """
        Figure which chunk of waveforms we should be
        slicing given our rank and world size
        """
        per_dev = x_per_y(abs(total), world_size)
        start = rank * per_dev
        stop = (rank + 1) * per_dev
        return start, stop

    def load_train_waveforms(self, f, world_size, rank):
        dataset = f["signals"]
        num_train = len(dataset)
        if not rank:
            self._logger.info(f"Training on {num_train} waveforms")

        start, stop = self.get_slice_bounds(num_train, world_size, rank)
        return self.load_signals(dataset, start, stop)

    def load_val_waveforms(self, f, world_size, rank):
        waveform_set = LigoWaveformSet.read(f)
        length = len(waveform_set.waveforms)

        if not rank:
            self._logger.info(f"Validating on {length} waveforms")
        stop, start = self.get_slice_bounds(length, world_size, rank)

        self._logger.info(f"Loading {start - stop} validation signals")
        start, stop = -start, -stop or None
        waveforms = torch.as_tensor(waveform_set.waveforms[start:stop])
        return waveforms

    def load_val_background(self, fnames: list[str]):
        self._logger.info("Loading validation background data")
        val_background = []
        for fname in fnames:
            segment = []
            with h5py.File(fname, "r") as f:
                for ifo in self.hparams.ifos:
                    segment.append(torch.Tensor(f[ifo][:]))
                val_background.append(torch.stack(segment))
        return val_background

    def build_transforms(self, sample_rate: float):
        """
        Helper utility in case we ever want to construct
        this dataset on its own.
        """
        window_length = self.hparams.kernel_length + self.hparams.fduration
        fftlength = self.hparams.fftlength or window_length
        self.psd_estimator = PsdEstimator(
            window_length,
            sample_rate,
            fftlength,
            fast=self.hparams.highpass is not None,
            average="median",
        )
        self.whitener = Whiten(
            self.hparams.fduration, sample_rate, self.hparams.highpass
        )
        self.projector = aug.WaveformProjector(
            self.hparams.ifos, sample_rate, self.hparams.highpass
        )

        self.sample_rate = sample_rate

    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)

        with h5py.File(self.train_fnames[0], "r") as f:
            sample_rate = 1 / f[self.hparams.ifos[0]].attrs["dx"]

        self._logger.info(f"Inferred sample rate {sample_rate}")

        # now define some of the augmentation transforms
        # that require sample rate information
        self._logger.info("Constructing sample rate dependent transforms")
        self.build_transforms(sample_rate)

        # load in our validation background up front and
        # compute which timeslides we'll do on this device
        # if we're doing distributed training so we'll know
        # which waveforms to subsample

        val_background = self.load_val_background(self.valid_fnames)
        self._logger.info(
            "Constructing validation timeslides from background segments "
            f"{' '.join(self.valid_fnames)}"
        )
        self.timeslides, self.valid_loader_length = get_timeslides(
            val_background,
            self.hparams.valid_livetime,
            self.sample_rate,
            self.sample_length,
            self.hparams.valid_stride,
            self.val_batch_size,
        )

        self._logger.info("Loading waveforms")
        with h5py.File(f"{self.data_dir}/train_waveforms.hdf5", "r") as f:
            cross, plus = self.load_train_waveforms(f, world_size, rank)
            self.waveform_sampler = aug.WaveformSampler(cross=cross, plus=plus)

        val_waveform_file = os.path.join(self.data_dir, "val_waveforms.hdf5")
        self.val_waveforms = self.load_val_waveforms(
            val_waveform_file, world_size, rank
        )
        self._logger.info("Initial dataloading complete")

    # ================================================ #
    # Utilities for doing augmentation/preprocessing
    # after tensors have been transferred to GPU
    # ================================================ #
    def _move_to_device(self, device):
        """
        This is dumb, but I genuinely cannot find a way
        to ensure that our transforms end up on the target
        device (see NOTE under self.get_local_device), so
        here's a lazy workaround to move our transforms once
        we encounter the first on-device tensor from our dataloaders.
        """
        if self._on_device:
            return
        for item in self.__dict__.values():
            if isinstance(item, torch.nn.Module):
                item.to(device)
        self._on_device = True

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            [X] = batch
            self._move_to_device(X)
            batch = self.augment(X)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch
            self._move_to_device(background)

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg)
        return batch

    def pad_waveforms(self, waveforms, kernel_size):
        """
        Add padding after a batch of waveforms to
        ensure that a uniformly sampled kernel will
        always be at least `self.pad_size` away
        from the end of the waveform (assumed to
        contain the coalescence) _after_ whitening.
        """
        filter_pad = int(self.hparams.fduration * self.sample_rate // 2)
        pad = kernel_size - filter_pad
        waveforms = waveforms[:, :, -pad - self.pad_size :]

        pad = kernel_size - self.pad_size
        waveforms = torch.nn.functional.pad(waveforms, [0, pad])
        return waveforms

    @torch.no_grad()
    def augment(self, X):
        """
        Override this in child classes to define
        application-specific augmentations
        """
        raise NotImplementedError

    # ================================================ #
    # Utilities for building train and validation dataloaders
    # ================================================ #
    @torch.no_grad()
    def build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Unfold a timeseries of background data
        into a batch of kernels, then inject
        multiple views of the provided signals
        into these timeseries. Whiten all tensors
        and return both the background and injected
        batches.
        """

        # TODO: in the same way we do inference, should we
        # use a longer PSD length and do the whitening
        # before we do the windowing to reduce compute?
        # The downside is this would mean doing true injections
        # for background data, which would require some fancy
        # footwork that I don't quite have time for.
        sample_size = int(self.sample_length * self.sample_rate)
        stride = int(self.hparams.valid_stride * self.sample_rate)
        background = unfold_windows(background, sample_size, stride=stride)

        X, psd = self.psd_estimator(background)
        X_bg = self.whitener(X, psd)

        # sometimes at the end of a segment, there won't be
        # enough background kernels and so we'll have to inject
        # our signals on overlapping data and ditch some at the end
        step = int(len(X) / len(signals))
        if not step:
            signals = signals[: len(X)]
        else:
            X = X[::step][: len(signals)]
            psd = psd[::step][: len(signals)]

        # create `num_view` instances of the injection on top of
        # the background, each showing a different, overlapping
        # portion of the signal
        kernel_size = X.size(-1)
        center = signals.size(-1) // 2

        step = kernel_size + 2 * self.pad_size
        step /= self.hparams.num_valid_views - 1
        X_inj = []
        for i in range(self.hparams.num_valid_views):
            start = center + self.pad_size - int(i * step)
            stop = start + kernel_size
            injected = X + signals[:, :, int(start) : int(stop)]
            injected = self.whitener(injected, psd)
            X_inj.append(injected)
        X_inj = torch.stack(X_inj)
        return X_bg, X_inj

    def val_dataloader(self) -> ZippedDataset:
        """
        Validation dataloader will iterate through batches
        in timeslides, returning both
        """
        background_dataset = pl.utilities.combined_loader.CombinedLoader(
            self.timeslides, mode="sequential"
        )
        iter(background_dataset)  # gives it a __len__ property

        # Figure out how many batches of background
        # we're going to go through, then batch the
        # signals so that they're spaced evenly
        # throughout all those batches.
        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // len(background_dataset) + 1
        signal_dataset = torch.utils.data.TensorDataset(self.val_waveforms)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return ZippedDataset(
            background_dataset, signal_loader, minimum=self.valid_loader_length
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        num_waveforms = self.waveform_sampler.num_waveforms

        waveforms_per_batch = (
            self.hparams.batch_size * self.hparams.waveform_prob
        )
        steps_per_epoch = int(4 * num_waveforms / waveforms_per_batch)

        # TODO: potentially introduce chunking here via
        # chunk_size/batches_per_chunk class args that
        # default to None
        dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=steps_per_epoch,
            coincident=False,
        )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        local_world_size = len(self.trainer.device_ids)
        num_workers = min(6, int(os.cpu_count() / local_world_size))
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, pin_memory=pin_memory
        )
        return dataloader
