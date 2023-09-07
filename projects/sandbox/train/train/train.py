import logging
from math import pi
from pathlib import Path
from typing import List, Optional

import torch
from train import utils as train_utils
from train import validation as valid_utils
from train.augmentations import SnrRescaler, SnrSampler
from train.augmentor import AframeBatchAugmentor, AugmentedDataset

from aframe.architectures.preprocessor import PsdEstimator
from aframe.logging import configure_logging
from aframe.trainer import trainify
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.distributions import Cosine, Uniform
from ml4gw.transforms import Whiten


# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to aframe.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from aframe.trainer.trainer.train to command line
# execution and parsing
@trainify
def main(
    # paths and environment args
    background_dir: Path,
    waveform_dataset: Path,
    outdir: Path,
    logdir: Path,
    ifos: List[str],
    # optimization args
    batch_size: int,
    snr_thresh: float,
    max_min_snr: float,
    max_snr: float,
    snr_alpha: float,
    snr_decay_steps: int,
    # data args
    sample_rate: float,
    kernel_length: float,
    psd_length: float,
    fduration: float,
    highpass: float,
    fftlength: Optional[float] = None,
    # augmentation args
    waveform_prob: float = 0.5,
    swap_frac: float = 0.0,
    mute_frac: float = 0.0,
    trigger_distance: float = 0,
    # validation args
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    num_valid_views: int = 5,
    max_fpr: float = 1e-3,
    valid_livetime: float = (3600 * 12),
    early_stop: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    # misc args
    device: str = "cpu",
    verbose: bool = False,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Prepare a dataset of background and pre-computed gravitational waves
    to train and validate an aframe architecture.

    Args:
        background_dir:
            Path to directory containing background segments for all ifos.
            The last of these segments will be used for validation if
            `valid_frac` is not `None`. Each file should be an HDF5 archive
            with a dataset containing strain data for each interferometer,
            labeled by the values in `ifos`.
        waveform_dataset:
            Path to file containing pre-computed gravitational
            wave polarization waveforms for binary black hole
            mergers. Should be an HDF5 archive with a `"signals"`
            dataset consisting of a tensor of shape
            `(num_waveforms, num_polarizations, waveform_size)`.
            At data-loading time, extrinsic parameters will be
            sampled for these events, which will be used to project
            them to interferometer responses which will then be
            injected into the corresponding channel with probability
            given by `waveform_prob`. Note that there is a nonzero
            likelihood that a waveform will be injected over
            a glitch. This will still be marked as a positive
            event in the training target.
        outdir:
            Directory to which validation artifacts will be saved,
            including best-performing model weights, training history,
            and an optional subdirectory for periodic checkpointing.
        logdir:
            Directory to which log files will be saved
        ifos:
            List of interferometers that there is background data for
            in each of the files of `background_dir`. Expected to be
            given by prefix; e.g. "H1" for Hanford
        batch_size:
            Number of samples over which to compute each gradient update
            during training.
        snr_thresh:
            During training, injected waveforms will be rescaled such that
            their SNRs follow a power law. The power law starts with some
            initial minimum (`max_min_snr`) which decreases over
            `snr_decay_steps` to `snr_thresh`. For the validation set,
            any waveforms with an SNR less than `snr_thresh` are scaled
            so that they have an SNR of `snr_thresh`.
        max_min_snr:
            The initial minimum value of the SNR distribution to which
            training injections will be scaled
        max_snr:
            The maximum of the SNR distribution to which training
            injections will be scaled
        snr_alpha:
            The exponent of the power law distribution. See
            `ml4gw.distributions.PowerLaw` for details.
        snr_decay_steps:
            The number of steps over which the minimum of the
            power law distribution will decrease from
            `max_min_snr` to `snr_thresh`
        sample_rate:
            The rate at which all relevant input data has
            been sampled, specified in Hz
        kernel_length:
            The length, in seconds, of each batch element to
            produce during iteration. This does not include
            the length of data removed after whitening.
        psd_length:
            The length, in seconds, of background to use for
            PSD calculation. This PSD will be used to whiten
            the next `fduration + kernel_length` seconds of
            data.
        fduration:
            Duration of the time domain filter used
            to whiten the data as a preprocessing step.
            Note that `fduration / 2` seconds worth of
            data will be cropped from both ends of the
            data being whitened before it is passed to
            the neural network.
        highpass:
            Minimum frequency over which to compute SNR values
            for waveform injection, in Hz. If left as `None`, the
            SNR will be computed over all frequency bins.
        fftlength:
            The length in seconds to use for FFT calculation when
            estimating the PSDs used to whiten the data. If left
            as `None`, the FFT length will be the same as the
            length of the unwhitened kernel.
        waveform_prob:
            The probability with which each sample in a batch
            will have a BBH waveform injected into its background.
        swap_frac:
            The fraction of kernels that will have a different
            injection in each interferometer. These kernels will
            be marked as background to teach the network that
            true signals are coherent.
        mute_frac:
            The fraction of kernels that will have an injection
            in only one interferometer. These kernels will be marked
            as background to teach the network that true signals
            are coincident.
        trigger_distance:
            The maximum length, in seconds, from the center of
            each waveform or glitch segment that a sampled
            kernel's edge can fall. The default value of `0`
            means that every kernel must contain the center
            of the corresponding segment (where the "trigger"
            or its equivalent is assumed to lie).
        valid_frac:
            Fraction of background, glitch, and waveform data
            to reserve for validation. Glitches and waveforms
            will be sampled once each, with the center of the
            segment in the center of the kernel, and either
            inserted or injected into windows of background.
        valid_stride:
            Distance, in seconds, between windows taken from
            the validation timeseries to pass to the network
            for validation.
        num_valid_views:
            During validation, `num_valid_view` copies of each
            kernel will be made, each with an injection in a
            different location within the kernel to make the
            network more robust against edge effects
        max_fpr:
            The false positive rate up to which the area under
            the ROC curve will be calculated during validation
        valid_livetime:
            The amount of background, in seconds, to create via
            time shifts during validation
        early_stop:
            Number of epochs without improvement of the validation
            metric before training should be terminated. If left
            as `None`, training will continue all the way through
            `max_epochs`.
        checkpoint_every:
            Indicates the frequency with which model weights
            should be checkpointed regardless of validation
            metric performance. If left as `None`, no
            checkpointing will occur and only the best
            performing weights will be saved.
        device:
            Device on which to perform training. Either `"cpu"`,
            `"cuda"`, or `"cuda:<device index>"` to train on a
            specific GPU.
        verbose:
            Controls log verbosity, with the default value of
            `False` logging at level `INFO`, and `True` logging
            at level `DEBUG`.
    """

    # make output dirs and configure logging file
    outdir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "train.log", verbose)
    if seed is not None:
        logging.info(f"Setting global seed to {seed}")
        train_utils.seed_everything(seed)

    # grab the names of the background files and determine the
    # length of data that will be handed to the preprocessor
    background_fnames = train_utils.get_background_fnames(background_dir)
    window_length = kernel_length + fduration
    sample_length = window_length + psd_length
    fftlength = fftlength or window_length

    # create objects that we'll use for whitening the data
    fast = highpass is not None
    psd_estimator = PsdEstimator(
        window_length, sample_rate, fftlength, fast=fast, average="median"
    )
    whitener = Whiten(fduration, sample_rate, highpass).to(device)

    # load the waveforms
    waveforms, valid_waveforms = train_utils.get_waveforms(
        waveform_dataset, ifos, sample_rate, valid_frac
    )

    if valid_waveforms is not None:
        valid_fname = background_fnames.pop(-1)
        logging.info(f"Loading validation segment {valid_fname}")
        valid_background = train_utils.get_background(valid_fname)

        # set up a tracker which will perform evaluation,
        # model selection, and checkpointing.
        tracker = valid_utils.LocalTracker(
            outdir,
            f"valid_auroc@{max_fpr:0.1e}",
            early_stop=early_stop,
            checkpoint_every=checkpoint_every,
        )

        # hard code some of these validation parameters
        # for now until it seems like they might need
        # some tuning
        validator = valid_utils.Validator(
            tracker,
            valid_background,
            valid_waveforms,
            psd_estimator=psd_estimator,
            whitener=whitener,
            snr_thresh=snr_thresh,
            highpass=highpass,
            sample_rate=sample_rate,
            stride=valid_stride,
            injection_stride=4,
            kernel_length=sample_length,
            batch_size=4 * batch_size,
            pool_length=4,
            integration_length=1,
            livetime=valid_livetime,
            shift=1,
            max_fpr=max_fpr,
            device=device,
            pad=-trigger_distance,
            num_views=num_valid_views,
        )
    else:
        validator = None

    # build some objects for augmenting waveform snrs
    waveform_duration = waveforms.shape[-1] / sample_rate
    rescaler = SnrRescaler(sample_rate, waveform_duration, highpass).to(device)
    snr_sampler = SnrSampler(
        max_min_snr=max_min_snr,
        min_min_snr=snr_thresh,
        max_snr=max_snr,
        alpha=snr_alpha,
        decay_steps=snr_decay_steps,
    )

    # now construct an object that will make
    # real-time augmentations to our training
    # data as its loaded
    cross, plus = waveforms.transpose(1, 0, 2)
    augmentor = AframeBatchAugmentor(
        ifos,
        sample_rate,
        waveform_prob,
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        psd_estimator=psd_estimator,
        whitener=whitener,
        trigger_distance=trigger_distance,
        mute_frac=mute_frac,
        swap_frac=swap_frac,
        snr=snr_sampler,
        rescaler=rescaler,
        invert_prob=0.5,
        reverse_prob=0.5,
        cross=cross,
        plus=plus,
    )
    augmentor = augmentor.to(device)

    # Create full training dataloader.
    # Use waveform dataset to dictate what
    # an "epoch" should be, adding a factor
    # to account for our sky parameter sampling
    # and to balance compute vs. validation resolution
    waveforms_per_batch = batch_size * waveform_prob
    batches_per_epoch = int(4 * len(waveforms) / waveforms_per_batch)

    train_dataset = Hdf5TimeSeriesDataset(
        fnames=background_fnames,
        channels=ifos,
        kernel_size=int(sample_length * sample_rate),
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        coincident=False,
    )

    kwargs = {}
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        kwargs["generator"] = g
        kwargs["worker_init_fn"] = train_utils.seed_worker
    if "cuda" in device:
        kwargs["pin_memory"] = True
        kwargs["pin_memory_device"] = device
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=4,
        **kwargs,
    )
    train_it = AugmentedDataset(train_dataloader, augmentor, device)

    return train_it, validator, None
