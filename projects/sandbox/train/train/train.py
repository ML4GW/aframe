from bbhnet.data import GlitchSampler, RandomWaveformDataset, WaveformSampler
from bbhnet.trainer.wrapper import trainify

# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from deepclean.trainer.trainer.train to command line
# execution and parsing

# note that this function is trivial:
# it simply just returns the data paths passed to it.
# however, future projects may have more complicated
# data discovery/generation processes.


@trainify
def main(
    glitch_dataset: str,
    signal_dataset: str,
    val_glitch_dataset: str,
    val_signal_dataset: str,
    hanford_background: str,
    livingston_background: str,
    val_hanford_background: str,
    val_livingston_background: str,
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
    """

    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity

    # initiate training glitch sampler
    train_glitch_sampler = GlitchSampler(glitch_dataset, device=device)

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

    # deterministic validation glitch sampler
    val_glitch_sampler = GlitchSampler(
        val_glitch_dataset,
        device=device,
    )

    # deterministic validation waveform sampler
    val_waveform_sampler = WaveformSampler(
        val_signal_dataset,
        sample_rate,
        min_snr,
        max_snr,
        highpass,
    )

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
        device,
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
        device,
    )

    return train_dataset, valid_dataset
