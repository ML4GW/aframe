import logging
from pathlib import Path
from typing import Optional

from train import utils as train_utils
from train import validation as valid_utils
from train.data_structures import (
    AframeInMemoryDataset,
    GlitchSampler,
    SnrRescaler,
    SnrSampler,
)

from aframe.architectures import Preprocessor
from aframe.logging import configure_logging
from aframe.trainer import trainify


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
    glitch_dataset: Path,
    waveform_dataset: Path,
    outdir: Path,
    logdir: Path,
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
    fduration: float,
    highpass: float,
    # augmentation args
    glitch_prob: float,
    waveform_prob: float,
    glitch_downweight: float,
    swap_frac: float = 0.0,
    mute_frac: float = 0.0,
    trigger_distance: float = 0,
    # validation args
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    max_fpr: float = 1e-3,
    valid_livetime: float = (3600 * 12),
    early_stop: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    # misc args
    device: str = "cpu",
    verbose: bool = False,
    **kwargs,
):
    """
    Prepare a dataset of background, pre-computed glitches,
    and pre-computed event waveforms to train and validate
    a aframe architecture.

    Args:
        background_dir:
            Path to directory containing background segments for all ifos.
            The first of such segments will be loaded in for training.
            Should be an HDF5 archive with a dataset
            containing strain data for each ifo labeled `"ifo"`. Must
            also contain an attribute `"t0"` indicating the start gpstime
            of the strain data.
        glitch_dataset:
            Path to file containing short segments of data
            with non-Gaussian noise transients. Should be
            an HDF5 archive with datasets `"<IFO ID>_glitches"`,
            where `IFO_ID` is the short ID for each interferometer
            used for training (H1 and L1 for now). These glitches
            will be used to randomly replace the corresponding
            interferometer channel during training with some
            probability given by `glitch_prob`. Note that the
            samples selected for insertion on each channel are
            sample independently, so glitches will be inserted
            into both channels with probability `glitch_prob**2`.
        waveform_dataset:
            Path to file containing pre-computed gravitational
            wave polarization waveforms for binary-blackhole
            merger events. Should be an HDF5 archive with a
            `"signals"` dataset consisting of a tensor of shape
            `(num_waveforms, num_polarizations, waveform_size)`.
            At data-loading time, extrinsic parameters will be
            sampled for these events, which will be used to project
            them to interferometer responses which will then be
            injected into the corresponding channel with probability
            given by `waveform_prob`. Note that the samples selected
            for injection will be chosen independently of those
            selected for glitch insertion, so there is a nonzero
            likelihood that a waveform will be injected over
            a glitch. This will still be marked as a positive
            event in the training target.
        glitch_prob:
            The probability with which each sample in a batch
            will have each of its interferometer channels
            replaced with a glitch from the `glitch_dataset`.
        waveform_prob:
            The probability with which each sample in a batch
            will have a BBH waveform injected into its background.
        sample_rate:
            The rate at which all relevant input data has
            been sampled.
        kernel_length:
            The length, in seconds, of each batch element
            to produce during iteration.
        batch_size:
            Number of samples to over which to compute each
            gradient update during training.
        mean_snr:
            Mean SNR of the log-normal distribution from which
            to sample SNR values for injected waveforms at
            data loading-time.
        std_snr:
            Standard deviation of the log-normal distribution
            from which to sample SNR values for injected waveforms
            at data loading-time.
        min_snr:
            Minimum SNR to use for SNR values for injected waveforms
            at data loading-time. Samples drawn from the log-normal
            SNR distribution below this value will be clipped to it.
            If left as `None`, all sampled SNRs will be used as-is.
        train_val_start:
            The gpstime that indicates the start
            of the contiguous training + validation background.
            This will be used to ensure glitches from the training set
            don't leak into the validation set when they are split.
        train_val_stop:
            The gpstime that indicates the end
            of the training background. This will be used to ensure
            glitches from the training set don't leak into the validation set
        highpass:
            Minimum frequency over which to compute SNR values
            for waveform injection, in Hz. If left as `None`, the
            SNR will be computed over all frequency bins.
        batches_per_epoch:
            Number of gradient updates in between each validation
            step. Implicitly controls the rate at which the learning
            can be decayed when training plateaus (since this is
            based on validation scores).
        fduration:
            Duration of the time domain filter used
            to whiten the data as a preprocessing step.
            Note that `fduration / 2` seconds worth of
            data will be cropped from both ends of the
            kernel of length `kernel_length` before passing
            it to the neural network.
        trigger_distance:
            The max length, in seconds, from the center of
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
        monitor_metric:
            Indicates whether model selection should be done
            using measurements of recall against performance
            on `"background"` or `"glitch"` data.
        threshold:
            Threshold of the indicated monitor metric against
            which to select the best-performing model. If
            `monitor_metric == "background"`, the allowed values
            are `[1, 2, 3, 4, 5]`. If `monitor_metric == "glitch"`,
            the allowed values are `[0.75, 0.9, 1]`.
        early_stop:
            Number of epochs without improvement in the indicated
            `monitor_metric` at the indicated `threshold` before
            training should be terminated. If left as `None`,
            training will continue all the way through `max_epochs`.
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

    # make out dir and configure logging file
    outdir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "train.log", verbose)

    # load background, infer ifos, and get start and end times
    # of the combined training + validation period
    logging.info("Loading background data")
    background, ifos, t0 = train_utils.get_background(background_dir)
    duration = background.shape[-1] / sample_rate
    logging.info(
        "Loaded {} seconds of data from ifos {}".format(
            duration, ", ".join(ifos)
        )
    )

    # load our waveforms and build some objects
    # for augmenting their snrs
    waveforms, valid_waveforms = train_utils.get_waveforms(
        waveform_dataset, ifos, sample_rate, valid_frac
    )
    waveform_duration = waveforms.shape[-1] / sample_rate
    rescaler = SnrRescaler(len(ifos), sample_rate, waveform_duration, highpass)
    snr_sampler = SnrSampler(
        max_min_snr, snr_thresh, max_snr, snr_alpha, snr_decay_steps
    )

    if valid_frac is not None:
        background, valid_background = train_utils.split(
            background, 1 - valid_frac, -1
        )
        tf = t0 + (1 - valid_frac) * duration

        # fit our rescaler here since we'll need its
        # background psd estimate to threshold
        logging.info("Rescaling validation waveforms")
        rescaler.fit(*background)
        valid_waveforms = train_utils.threshold_snrs(
            valid_waveforms,
            snr_thresh,
            sample_rate,
            rescaler.background,
            highpass,
        )

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
            sample_rate,
            stride=valid_stride,
            injection_stride=4,
            kernel_length=kernel_length,
            batch_size=4 * batch_size,
            pool_length=4,
            integration_length=1,
            livetime=valid_livetime,
            shift=1,
            max_fpr=max_fpr,
            device=device,
        )
    else:
        tf = t0 + duration
        rescaler.fit(*background)
        validator = None
    rescaler = rescaler.to(device)

    # load our glitches and build an object
    # for randomly sampling from them
    logging.info("Loading glitches")
    glitches = train_utils.get_glitches(glitch_dataset, ifos, tf)
    for ifo, dset in glitches.items():
        logging.info(f"{len(dset)} glitches from {ifo}")
    glitch_sampler = GlitchSampler(
        prob=glitch_prob,
        max_offset=int(trigger_distance * sample_rate),
        **glitches,
    )

    # now construct an object that will make
    # real-time augmentations to our training
    # data as its loaded
    augmentor = train_utils.get_augmentor(
        ifos,
        sample_rate,
        glitch_sampler,
        waveforms,
        waveform_prob,
        snr_sampler,
        rescaler,
        mute_frac,
        swap_frac,
        glitch_downweight,
        trigger_distance,
    )
    augmentor = augmentor.to(device)

    # Create full training dataloader.
    # Use waveform dataset to dictate what
    # an "epoch" should be
    waveforms_per_batch = batch_size * waveform_prob
    batches_per_epoch = len(waveforms) / waveforms_per_batch
    train_dataset = AframeInMemoryDataset(
        background,
        int(kernel_length * sample_rate),
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        preprocessor=augmentor,
        coincident=False,
        shuffle=True,
        device=device,
    )

    preprocessor = Preprocessor(
        len(ifos), sample_rate=sample_rate, fduration=fduration
    )

    # fit the whitening module to the background then
    # move eveyrthing to the desired device
    preprocessor.whitener.fit(kernel_length, *background)
    preprocessor.whitener.to(device)

    return train_dataset, validator, preprocessor
