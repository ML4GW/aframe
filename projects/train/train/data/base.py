import glob
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import lightning.pytorch as pl
import torch
from ml4gw.augmentations import SignalInverter, SignalReverser
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import unfold_windows

from train import augmentations as aug
from train.data.utils import fs as fs_utils
from train.metrics import get_timeslides
from train.data.waveforms import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformLoader,
    WaveformSampler,
)
from utils.preprocessing import PsdEstimator

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution
TransformedDist = torch.distributions.TransformedDistribution


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
    """
    Base LightningDataModule for loading data to train Aframe models.
    Subclasses must define the `inject` method
    which encodes how background strain,
    cross/plus polarizations and parameters
    are processed before being passed to a model

    Args:
        background_dir:
            Path to the directory containing the training background.
            If this is a s3 bucket, it will be downloaded
            to a local directory.
        waveforms_dir:
            Path to the directory containing the training waveforms.
            If this is a s3 bucket, it will be downloaded
            to a local directory.
        ifos:
            List of interferometers to use for training.
        sample_rate:
            Sample rate in Hz of the training data.
        dec:
            The distribution of declinations to sample from
        psi:
            The distribution of polarization angles to sample from
        phi:
            The distribution of "right ascensions" to sample from
        batches_per_epoch:
            Number of batches per epoch.
        num_files_per_batch:
            Number of background files to sample from each batch.
        waveform_sampler:
            `WaveformSampler` object that produces waveforms and parameters
            for training and validation. See `train.data.waveforms.sampler`
            for methods this object should define.
        max_num_workers:
            Maximum number of workers to assign to each
            training dataloader.
        batch_size:
            Batch size for training.
        kernel_length:
            Length in seconds of the time-domain kernel
            passed to the model after whitening.
        fduration:
            The length of the whitening filter's impulse
            response, in seconds. `fduration / 2` seconds
            worth of data will be cropped from the edges
            of the whitened timeseries.
        psd_length:
            Length in seconds of the PSD used for whitening.
        waveform_prob:
            Probability that a batch element will contain a waveform.
        left_pad:
            Length in seconds of the minimum gap between the left
            edge of whitened kernel and coalescence point of the signal.
        right_pad:
            Length in seconds of the minimum gap between the right
            edge of whitened kernel and coalescence point of the signal.
        fftlength:
            Length in seconds of the FFT used for PSD estimation.
            If `None`, will use the length of the kernel + fduration.
        highpass:
            Highpass filter frequency in Hz.
        lowpass:
            Lowpass filter frequency in Hz.
        snr_sampler:
            A callable that samples SNRs for the injected signals.
            If `None`, SNRs will be left unchanged.
        valid_stride:
            Stride in seconds for the validation timeslides.
            If `None`, will use `kernel_length + fduration`.
        num_valid_views:
            Number of views to inject into the validation data.
            Each view will contain the same signal at a different
            position in the background.
        min_valid_duration:
            Minimum duration in seconds of the validation data.
        valid_livetime:
            Total livetime in seconds of the validation data
            to be generated via timeslides.
        chunks_per_epoch:
            Number of chunks of waveforms to load from disk
            each epoch. Not used if generating waveforms
            during training.
        chunk_size:
            Number of waveforms to load in each chunk.
            Not used if generating waveforms during training.
        verbose:
            Whether to log debug information during training.
    """

    def __init__(
        self,
        # data loading args
        background_dir: str,
        waveforms_dir: str,
        ifos: Sequence[str],
        sample_rate: float,
        dec: Distribution,
        psi: Distribution,
        phi: Distribution,
        batches_per_epoch: int,
        num_files_per_batch: int,
        waveform_sampler: WaveformSampler,
        # preprocessing args
        batch_size: int,
        kernel_length: float,
        fduration: float,
        psd_length: float,
        # augmentation args
        waveform_prob: float = 1,
        left_pad: float = 0,
        right_pad: float = 0,
        fftlength: Optional[float] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        snr_sampler: Optional[
            Union[TransformedDist, Callable[[int], Tensor]]
        ] = None,
        # validation args
        valid_stride: Optional[float] = None,
        num_valid_views: int = 4,
        min_valid_duration: float = 15000,
        valid_livetime: float = (3600 * 12),
        max_num_workers: int = 6,
        # dataloading args
        chunks_per_epoch: int = 1,
        chunk_size: int = 10000,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.init_logging(verbose)
        self.num_ifos = len(ifos)
        self.max_num_workers = max_num_workers
        self.save_hyperparameters(ignore=["waveform_sampler"])

        # Set up some of our data augmentation modules
        self.inverter = SignalInverter(0.5)
        self.reverser = SignalReverser(0.5)

        # these are modules that require our data to be
        # downloaded first, either for loading signals
        # or to infer sample rate, so wait to construct
        # them until self.setup
        self.whitener = None
        self.projector = None
        self.psd_estimator = None
        self._on_device = False

        self.dec, self.psi, self.phi = dec, psi, phi
        self.waveform_sampler = waveform_sampler
        # If we're using a `WaveformLoader`, we're loading
        # training waveforms from disk, so have a flag tp
        # indicate that
        self.waveforms_from_disk = isinstance(waveform_sampler, WaveformLoader)
        self.snr_sampler = snr_sampler

        # generate our local node data directory
        # if our specified data source is remote
        self.background_dir = fs_utils.get_data_dir(
            self.hparams.background_dir
        )
        self.waveforms_dir = fs_utils.get_data_dir(self.hparams.waveforms_dir)
        self.verbose = verbose

    # ================================================ #
    # Distribution utilities
    # ================================================ #

    def init_logging(self, verbose: bool):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.DEBUG if verbose else logging.INFO,
            stream=sys.stdout,
        )

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
        return logger

    # ================================================ #
    # Re-paramterizing some attributes
    # ================================================ #

    @property
    def batches_per_epoch(self) -> int:
        world_size, _ = self.get_world_size_and_rank()
        return self.hparams.batches_per_epoch // world_size

    # TODO: can probably make this CLI configurable at some point
    @property
    def psd_window(self):
        """
        Window applied to fft segments during psd estimation
        If left as `None` `SpectralDensity` will use a hann window
        """
        return None

    @property
    def sample_length(self) -> float:
        """Length of samples generated by datasets in seconds"""
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

    @property
    def filter_size(self) -> int:
        """
        Length of the time-domain whitening filter in samples
        """
        return int(self.hparams.fduration * self.hparams.sample_rate)

    @property
    def left_pad_size(self) -> int:
        """
        Minimum numer of samples that the defining point of the
        signal will be from the left edge of the _whitened_ kernel.
        """
        return int(self.hparams.left_pad * self.hparams.sample_rate)

    @property
    def right_pad_size(self) -> int:
        """
        Minimum number of samples that the defining point of the
        signal will be from the left edge of the _whitened_ kernel
        """
        return int(self.hparams.right_pad * self.hparams.sample_rate)

    def train_val_split(self) -> tuple[Sequence[str], Sequence[str]]:
        fnames = glob.glob(f"{self.background_dir}/background/*.hdf5")
        fnames = sorted([Path(fname) for fname in fnames])
        durations = [int(fname.stem.split("-")[-1]) for fname in fnames]
        valid_fnames = []
        valid_duration = 0
        while valid_duration < self.hparams.min_valid_duration:
            fname, duration = fnames.pop(-1), durations.pop(-1)
            valid_duration += duration
            valid_fnames.append(str(fname))

        train_fnames = list(set(fnames) - set(valid_fnames))
        return train_fnames, valid_fnames

    @property
    def val_batch_size(self):
        """Use larger batch sizes when we don't need gradients."""
        return int(1 * self.hparams.batch_size)

    @property
    def num_workers(self):
        local_world_size = len(self.trainer.device_ids)
        return min(
            self.max_num_workers, int(os.cpu_count() / local_world_size)
        )

    # ================================================ #
    # Utilities for initial data loading and preparation
    # ================================================ #

    def prepare_data(self):
        """
        Download s3 data if it doesn't exist.
        """
        logger = logging.getLogger("AframeDataset")
        bucket, _ = fs_utils.split_data_dir(self.hparams.background_dir)
        if bucket is None:
            return
        logger.info(
            "Downloading background data from S3 bucket {} to {}".format(
                bucket, self.background_dir
            )
        )
        fs_utils.download_training_data(bucket, self.background_dir)

        bucket, _ = fs_utils.split_data_dir(self.hparams.waveforms_dir)
        if bucket is None:
            return
        logger.info(
            "Downloading waveform data from S3 bucket {} to {}".format(
                bucket, self.waveforms_dir
            )
        )
        fs_utils.download_training_data(bucket, self.waveforms_dir)

    def slice_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Slice waveforms to the correct length depending on
        requested left and right padding
        """
        signal_idx = waveforms.shape[-1] - int(
            self.waveform_sampler.right_pad * self.hparams.sample_rate
        )
        kernel_size = int(
            self.hparams.kernel_length * self.hparams.sample_rate
        )

        if kernel_size < self.left_pad_size + self.right_pad_size:
            raise ValueError(
                f"Kernel size ({kernel_size}) cannot be less than total "
                f"padding ({self.left_pad_size} + {self.right_pad_size})"
            )

        signal_start = signal_idx - (kernel_size - self.right_pad_size)
        signal_start -= self.filter_size // 2

        signal_stop = signal_idx + (kernel_size - self.left_pad_size)
        signal_stop += self.filter_size // 2

        # If signal_start is less than 0, add padding on the left
        left_pad = -1 * min(signal_start, 0)
        # If signal_stop is larger than the dataset, add padding on the right
        right_pad = max(signal_stop - waveforms.shape[-1], 0)

        waveforms = torch.nn.functional.pad(waveforms, [left_pad, right_pad])
        waveforms = waveforms[..., signal_start:signal_stop]

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

    def transforms_to_device(self):
        """
        Move all `torch.nn.Modules` to the local device
        """
        for item in self.__dict__.values():
            if isinstance(item, torch.nn.Module):
                item.to(self.device)

    def build_transforms(self):
        """
        Helper utility in case we ever want to construct
        this dataset on its own.
        """
        window_length = self.hparams.kernel_length + self.hparams.fduration
        fftlength = self.hparams.fftlength or window_length
        self.psd_estimator = PsdEstimator(
            window_length,
            self.hparams.sample_rate,
            fftlength,
            window=self.psd_window,
            fast=self.hparams.highpass is not None,
            average="median",
        )
        self.whitener = Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        self.projector = aug.WaveformProjector(
            self.hparams.ifos,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )

    def sample_extrinsic(self, X: torch.Tensor):
        """
        Sample extrinsic parameters used to project waveforms
        """
        N = len(X)
        dec = self.dec.sample((N,)).to(X.device)
        psi = self.psi.sample((N,)).to(X.device)
        phi = self.phi.sample((N,)).to(X.device)
        return dec, psi, phi

    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        self.train_fnames, self.valid_fnames = self.train_val_split()

        with h5py.File(self.train_fnames[0], "r") as f:
            sample_rate = 1 / f[self.hparams.ifos[0]].attrs["dx"]
            if not sample_rate == self.hparams.sample_rate:
                raise ValueError(
                    f"Specified sample rate is {self.hparams.sample_rate} "
                    f"but background data is sampled at {sample_rate}"
                )

        self._logger.info(f"Validated sample rate {sample_rate}")

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
            self.hparams.sample_rate,
            self.sample_length,
            self.hparams.valid_stride,
            self.val_batch_size,
        )

        self.val_waveforms = self.waveform_sampler.get_val_waveforms(
            world_size, rank
        )
        self.waveform_sampler.get_train_waveforms(
            world_size, rank, self.device
        )
        self._logger.info("Initial dataloading complete")

        # now define some of the augmentation transforms
        # that require sample rate information
        self._logger.info("Constructing sample rate dependent transforms")
        self.build_transforms()
        self.transforms_to_device()

    # ================================================ #
    # Utilities for doing augmentation/preprocessing
    # after tensors have been transferred to GPU
    # ================================================ #
    @property
    def device(self):
        """Return the device of the associated lightning module"""
        return self.trainer.lightning_module.device

    def on_before_batch_transfer(self, batch, _):
        """
        Slice loaded waveforms before sending to device
        if not generating waveforms during training
        """
        # TODO: maybe pass indices as argument to
        # waveform loader to reduce quantity of data
        # we need to load
        if self.trainer.training and self.waveforms_from_disk:
            X, waveforms = batch
            waveforms = self.slice_waveforms(waveforms)
            batch = X, waveforms
        return batch

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
            if self.waveforms_from_disk:
                [batch], waveforms = batch
                batch = self.inject(batch, waveforms)
            else:
                [batch] = batch
                batch = self.inject(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg)
        return batch

    @torch.no_grad()
    def inject(self, X):
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
        self,
        background: Tensor,
        signals: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Unfold a timeseries of background data
        into a batch of kernels, then inject
        multiple views of the provided signals
        into these timeseries.

        Args:
            background: A tensor of background data
            signals: A tensor of signals to inject

        Returns:
            raw strain background kernels, injected kernels, and psds
        """

        # unfold the background data into kernels
        sample_size = int(self.sample_length * self.hparams.sample_rate)
        stride = int(self.hparams.valid_stride * self.hparams.sample_rate)
        background = unfold_windows(background, sample_size, stride=stride)

        # split data into kernel and psd data and estimate psd
        X, psd = self.psd_estimator(background)

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
        signal_idx = signals.shape[-1] - int(
            self.waveform_sampler.right_pad * self.hparams.sample_rate
        )
        max_start = int(
            signal_idx - self.left_pad_size - self.filter_size // 2
        )
        max_stop = max_start + kernel_size
        pad = max_stop - signals.size(-1)
        if pad > 0:
            signals = torch.nn.functional.pad(signals, [0, pad])

        step = (
            kernel_size
            - self.left_pad_size
            - self.right_pad_size
            - self.filter_size
        )
        step /= self.hparams.num_valid_views - 1
        X_inj = []
        for i in range(self.hparams.num_valid_views):
            start = max_start - int(i * step)
            stop = start + kernel_size
            injected = X + signals[:, :, int(start) : int(stop)]
            X_inj.append(injected)
        X_inj = torch.stack(X_inj)

        return X, X_inj, psd

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
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        signal_dataset = torch.utils.data.TensorDataset(self.val_waveforms)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        dataset = ZippedDataset(
            background_dataset,
            signal_loader,
            minimum=min(self.valid_loader_length, len(signal_loader)),
        )
        return dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # build our strain dataset and dataloader
        dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            num_files_per_batch=self.hparams.num_files_per_batch,
        )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        self._logger.debug(
            f"Using {self.num_workers} workers for strain data loading"
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        # If we're not loading waveforms from disk, just return
        # the background dataloader
        if not self.waveforms_from_disk:
            return dataloader

        # build iterator for waveform loading
        # that will load chunks of waveforms
        # to be sampled from
        waveform_loader = Hdf5WaveformLoader(
            self.waveform_sampler.training_waveform_files,
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
            channels=["cross", "plus"],
            path="waveforms",
        )
        # calculate how many batches we'll sample from each chunk
        # based on requested chunks per epoch and batches per epoch
        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = (
            int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1
        )
        self._logger.info(
            f"Training on pool of {waveform_loader.total} waveforms. "
            f"Sampling {batches_per_chunk} batches per chunk "
            f"from {self.hparams.chunks_per_epoch} chunks "
            f"of size {self.hparams.chunk_size} each epoch"
        )

        # multiprocess waveform chunk loader
        # so we don't have to wait for waveforms
        waveform_loader = torch.utils.data.DataLoader(
            waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )

        # build a dataset that will sample from
        # iterator of chunks of waveforms
        waveform_dataset = ChunkedWaveformDataset(
            waveform_loader,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )

        return ZippedDataset(dataloader, waveform_dataset)
