from pathlib import Path
from typing import Optional

import torch

from bbhnet.data import (
    DeterministicWaveformDataset,
    GlitchSampler,
    RandomWaveformDataset,
    WaveformSampler,
)
from bbhnet.data.transforms import WhiteningTransform
from bbhnet.logging import configure_logging
from bbhnet.trainer import trainify

# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from bbhnet.trainer.trainer.train to command line
# execution and parsing


@trainify
def main(
    glitch_dataset: str,
    signal_dataset: str,
    hanford_background: str,
    livingston_background: str,
    waveform_frac: float,
    glitch_frac: float,
    kernel_length: float,
    min_snr: float,
    max_snr: float,
    highpass: float,
    sample_rate: float,
    batch_size: int,
    batches_per_epoch: int,
    device: str,
    outdir: Path,
    logdir: Path,
    fduration: Optional[float] = None,
    trigger_distance_size: float = 0,
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    verbose: bool = False,
    **kwargs
):
    """
    waveform_frac:
        The fraction of waveforms in each batch
    glitch_frac:
        The fraction of glitches in each batch
    sample_rate:
        The rate at which all relevant input data has
        been sampled
    kernel_length:
        The length, in seconds, of each batch element
        to produce during iteration.
    min_snr:
        Minimum SNR value for sampled waveforms.
    max_snr:
        Maximum SNR value for sampled waveforms.
    highpass:
        Frequencies above which to keep
    batch_size:
        Number of samples to produce during at each
        iteration
    batches_per_epoch:
        The number of batches to produce before raising
        a `StopIteration` while iteratingkernel_length:
    fduration:
        duration of the time domain filter used
        to whiten the data (If using WhiteningTransform).
        Note that fduration / 2 seconds will be cropped from
        both ends of kernel_length
    trigger_distance_size:

    """

    # make out dir and configure logging file
    outdir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir / "train.log", verbose)

    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity

    if valid_frac is not None:
        frac = 1 - valid_frac
    else:
        frac = None

    # initiate training glitch sampler
    train_glitch_sampler = GlitchSampler(glitch_dataset, frac=frac)

    # initiate training waveform sampler
    train_waveform_sampler = WaveformSampler(
        signal_dataset,
        sample_rate,
        min_snr=min_snr,
        max_snr=max_snr,
        highpass=highpass,
        frac=frac,
    )

    # create full training dataloader
    train_dataset = RandomWaveformDataset(
        hanford_background,
        livingston_background,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        waveform_sampler=train_waveform_sampler,
        waveform_frac=waveform_frac,
        glitch_sampler=train_glitch_sampler,
        glitch_frac=glitch_frac,
        trigger_distance=trigger_distance_size,
        frac=frac,
    )

    # TODO: hard-coding num_ifos into preprocessor. Should
    # we just expose this as an arg? How will this fit in
    # to the broader-generalization scheme?
    preprocessor = WhiteningTransform(
        2, sample_rate, kernel_length, highpass=highpass, fduration=fduration
    )

    # TODO: make this a `train_dataset.background` `@property`?
    background = torch.stack(
        [train_dataset.hanford_background, train_dataset.livingston_background]
    )
    preprocessor.fit(background)

    # deterministic validation glitch sampler
    if valid_frac is not None:
        if valid_stride is None:
            raise ValueError(
                "Must specify a validation stride if "
                "specifying a validation fraction"
            )

        val_glitch_sampler = GlitchSampler(
            glitch_dataset, deterministic=True, frac=-valid_frac
        )

        # deterministic validation waveform sampler
        val_waveform_sampler = WaveformSampler(
            signal_dataset,
            sample_rate,
            min_snr=min_snr,
            max_snr=max_snr,
            highpass=highpass,
            deterministic=True,
            frac=-valid_frac,
        )
        val_waveform_sampler.fit(
            H1=train_dataset.hanford_background,
            L1=train_dataset.livingston_background,
        )

        # create full validation dataloader
        valid_dataset = DeterministicWaveformDataset(
            hanford_background,
            livingston_background,
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            stride=valid_stride,
            batch_size=4 * batch_size,
            waveform_sampler=val_waveform_sampler,
            glitch_sampler=val_glitch_sampler,
            offset=0,
            frac=-valid_frac,
        )
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, preprocessor
