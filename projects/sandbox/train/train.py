from pathlib import Path
from typing import Optional

import torch

from bbhnet.data import GlitchSampler, RandomWaveformDataset, WaveformSampler
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
    val_glitch_dataset: str = None,
    val_signal_dataset: str = None,
    val_hanford_background: str = None,
    val_livingston_background: str = None,
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

    # TODO: definitely a cleaner way to set validation flag
    # if validation files are all passed, set validate bool to true
    validation_files = (
        val_glitch_dataset,
        val_signal_dataset,
        val_hanford_background,
        val_livingston_background,
    )
    validate = all([f is not None for f in validation_files])

    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity

    # initiate training glitch sampler
    train_glitch_sampler = GlitchSampler(glitch_dataset)

    # initiate training waveform sampler
    train_waveform_sampler = WaveformSampler(
        signal_dataset,
        sample_rate,
        min_snr,
        max_snr,
        highpass,
    )

    # TODO: incorporate Erics deterministic
    # sampling into validation loaders

    # create full training dataloader
    train_dataset = RandomWaveformDataset(
        hanford_background,
        livingston_background,
        kernel_length,
        sample_rate,
        batch_size,
        batches_per_epoch,
        train_waveform_sampler,
        waveform_frac,
        train_glitch_sampler,
        glitch_frac,
        trigger_distance_size,
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
    if validate:
        val_glitch_sampler = GlitchSampler(
            val_glitch_dataset,
        )

        # deterministic validation waveform sampler
        val_waveform_sampler = WaveformSampler(
            val_signal_dataset,
            sample_rate,
            min_snr,
            max_snr,
            highpass,
        )

        # create full validation dataloader
        valid_dataset = RandomWaveformDataset(
            val_hanford_background,
            val_livingston_background,
            kernel_length,
            sample_rate,
            batch_size,
            batches_per_epoch,
            val_waveform_sampler,
            waveform_frac,
            val_glitch_sampler,
            glitch_frac,
        )
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, preprocessor
