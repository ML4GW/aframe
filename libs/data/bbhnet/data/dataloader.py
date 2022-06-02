from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from gwpy.frequencyseries import FrequencySeries
from gwpy.signal.filter_design import fir_from_transfer
from gwpy.timeseries import TimeSeries

from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.utils import sample_kernels
from bbhnet.data.waveform_sampler import WaveformSampler

DEFAULT_FFTLENGTH = 2


def _build_time_domain_filter(
    asd: FrequencySeries,
    sample_rate: float,
    kernel_length: float,
    fftlength: float = DEFAULT_FFTLENGTH,
) -> np.ndarray:
    """
    Create a time-domain filter for whitening using the
    indicated timeseries as background. Replicating the
    behavior of `gwpy.timeseries.TimeSeries.whiten`.
    Args:
        timeseries:
            Background timeseries whose spectrum to
            use for whitening
        sample_rate:
            Data rate of timeseries
        kernel_length:
            Length of data that will eventually be used
            to be whitened by this filter
    Returns:
        Time domain filter to convolve with sampled
        training data
    """

    asd = asd.interpolate(1 / kernel_length)
    ntaps = int(fftlength * sample_rate)

    tdf = fir_from_transfer(
        1 / asd.value, ntaps=ntaps, window="hanning", ncorner=0
    )
    return tdf


def _load_background(
    background_file: str,
    sample_rate: float,
    device: str,
    fftlength: float = DEFAULT_FFTLENGTH,
):
    # TODO: maybe these are gwf and we resample?
    with h5py.File(background_file, "r") as f:
        background = f["hoft"][:]

        # grab the timestamps from the dataset for geocent sampling
        t0 = f["t0"][()]

    # build the asd for building the time domain filter
    ts = TimeSeries(background, dt=1 / sample_rate)
    asd = ts.asd(fftlength=fftlength, window="hanning", method="median")

    # move everything onto the GPU up front so that
    # we don't have to pay for transfer time later.
    # If our datasets are on the scale of ~GBs this
    # shouldn't be a problem, esp. for the current
    # size of BBHNet
    background = torch.Tensor(background).to(device)
    return background, asd, t0


class RandomWaveformDataset:
    def __init__(
        self,
        hanford_background: str,
        livingston_background: str,
        kernel_length: float,
        sample_rate: float,
        batch_size: int,
        batches_per_epoch: int,
        waveform_sampler: Optional[WaveformSampler] = None,
        waveform_frac: float = 0,
        glitch_sampler: Union[GlitchSampler, str, None] = None,
        glitch_frac: float = 0,
        device: torch.device = "cuda",
    ) -> None:
        """Iterable dataset which can sample and inject auxiliary data

        Iterable dataset for use with torch.data.DataLoader which
        generates tensors of background data from the two LIGO
        interferometers. Optionally can inject simulated waveforms
        and insert real glitch data which are sampled from HDF5
        datasets.
        Background data is sample uniformly and independently for
        both interferometers, simulating arbitrary time-shifts.
        The cost of this is that we abandon the traditional notion
        of an "epoch" as "one full pass through the dataset", since
        the sampling makes no attempt to exclude kernels which may
        have been sampled recently. As such, the `batches_per_epoch`
        kwarg is used to determine how many batches to produce
        before to raising a `StopIteration` to move on to tasks
        like validation.
        Args:
            hanford_background:
                Path to HDF5 file containing background data for
                the Hanford interferometer under the dataset
                key `"hoft"`. Assumed to be sampled at rate
                `sample_rate`.
            livingston_background:
                Path to HDF5 file containing background data for
                the Livingston interferometer under the datset
                key `"hoft"`. Assumed to be sampled at rate
                `sample_rate`. Must contain the same amount of
                data as `hanford_background`.
            kernel_length:
                The length, in seconds, of each batch element
                to produce during iteration.
            sample_rate:
                The rate at which all relevant input data has
                been sampled
            batch_size:
                Number of samples to produce during at each
                iteration
            batches_per_epoch:
                The number of batches to produce before raising
                a `StopIteration` while iterating
            waveform_sampler:
                An object with `.fit` and `.sample` attributes for
                sampling interferometer responses to raw waveforms
                for injection into background data. If left as `None`,
                no injection will take place at data loading time and
                `waveform_frac` must be set to 0.
            waveform_frac:
                The fraction of each batch that should consist
                of injected waveforms, and be marked with a
                `1.` in the target tensor produced during iteration.
            min_snr:
                Minimum SNR value for sampled waveforms. Must be
                specified if `waveform_dataset` is not `None`.
            max_snr:
                Maximum SNR value for sampled waveforms. Must be
                specified if `waveform_dataset` is not `None`.
            glitch_sampler:
                Object for sampling glitches for insertion (not injection)
                into background samples at data loading time. Must
                include a `.sample` method. If passed a string, should be
                a path to an HDF5 file containing glitches from each
                interferometer contained in `"H1_glitches"` and
                `"L1_glitches"` dataset keys. Data should be
                sampled at rate `sample_rate`. Glitch triggers
                should sit in the middle of the time axis of
                the array containing the glitch timeseries.
                Glitches will be randomly sampled and inserted in
                place of the corresponding interferometer channel
                background at data-loading time. If left as `None`,
                no glitches will be inserted at data loading time
                and `glitch_frac` should be 0.
            glitch_frac:
                The fraction of each batch that should consist
                of inserted glitches, marked as `0.` in the
                target tensor produced during iteration
            device:
                The device on which to host all the relevant
                torch tensors.
        """

        # sanity check our fractions
        assert 0 <= waveform_frac <= 1
        assert 0 <= glitch_frac <= 1

        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.device = device

        # load in the background data and build time-domain
        # filters for whitening using them
        self.hanford_background, hanford_asd, t0 = _load_background(
            hanford_background, sample_rate, device
        )
        self.livingston_background, livingston_asd, t0 = _load_background(
            livingston_background, sample_rate, device
        )

        assert len(self.hanford_background) == len(self.livingston_background)
        tf = t0 + len(self.hanford_background) / sample_rate

        # build time-domain filters using the background
        # asd so that we can perform whitening using a
        # grouped 1d convolution at data-loading time
        hanford_tdf = _build_time_domain_filter(
            hanford_asd, sample_rate=sample_rate, kernel_length=kernel_length
        )
        livingston_tdf = _build_time_domain_filter(
            livingston_asd,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
        )

        # move the filters to a single torch Tensor
        # on the desired device, adding a dummy dimension
        # in the middle for compatability with Torch's
        # conv op's expectations
        whitening_filter = np.stack([hanford_tdf, livingston_tdf])
        self.whitening_filter = torch.Tensor(whitening_filter[:, None]).to(
            device
        )

        # create a window for applying the time domain filter
        # at data loading time. Since the filter will have the
        # same size as the timeseries at data loading time,
        # we can use the window as-is without having to apply
        # it just to the edges of the timeseries
        self.whitening_window = torch.hann_window(
            whitening_filter.shape[-1], device=device
        )

        # if we specified a waveform sampler, fit its snr
        # computation to the given background asd
        if waveform_sampler is not None:
            assert waveform_frac > 0

            # give the asds channel names so that the waveform
            # sampler knows which ifo responses to calculate
            if hanford_asd.channel is None:
                hanford_asd.channel = "H1:STRAIN"
            if livingston_asd.channel is None:
                livingston_asd.channel = "L1:STRAIN"

            # now fit the the waveform_sampler's background_asd
            # attribute to the given asds for the snr computation
            waveform_sampler.fit(t0, tf, hanford_asd, livingston_asd)

            # assign our attributes
            self.waveform_sampler = waveform_sampler
            self.num_waveforms = max(1, int(waveform_frac * batch_size))
        else:
            # likewise, ensure that we didn't indicate that
            # we expected any waveforms in the batch
            assert waveform_frac == 0
            self.num_waveforms = 0
            self.waveform_sampler = waveform_sampler

        # load in any glitches if we specified them
        if glitch_sampler is not None:
            # if we specified glitches, make sure we're
            # actually planning on using them
            assert glitch_frac > 0
            self.num_glitches = max(1, int(glitch_frac * batch_size))

            if isinstance(glitch_sampler, (str, Path)):
                glitch_sampler = GlitchSampler(glitch_sampler, device)
            self.glitch_sampler = glitch_sampler
        else:
            # likewise, ensure that we didn't indicate that
            # we expected any glitches in the batch
            assert glitch_frac == 0
            self.num_glitches = 0
            self.glitch_sampler = None

        # make sure that we have at least _some_
        # pure background in each batch
        assert (self.num_waveforms + self.num_glitches) < batch_size

    def sample_from_background(self):  # , independent: bool = True):
        """Sample a batch of kernels from the background data

        Randomly sample kernels from the interferometer
        background timeseries in a uniform manner. Removing
        the `independent` kwarg for now to leverage the
        more general sample_kernels function.

        TODO: figure out how best to generalize sample_kernels
            to make use of pre-sampled idx
        """

        hanford_kernels = sample_kernels(
            self.hanford_background, self.kernel_size, self.batch_size
        )
        livingston_kernels = sample_kernels(
            self.livingston_background, self.kernel_size, self.batch_size
        )

        # interweave these kernels along the 0th axis so that
        # a reshape puts them in the right channel dimension
        kernels = zip(hanford_kernels, livingston_kernels)
        kernels = [i for j in kernels for i in j]
        kernels = torch.stack(kernels, dim=0)
        return kernels.reshape(self.batch_size, 2, -1)

    def whiten(self, X: torch.Tensor) -> torch.Tensor:
        """Detrend and time-domain filter an array
        Use the time-domain filters built from the
        background data used to initialize the dataset
        to whiten an array of data.
        """

        # do a constant detrend along the time axis,
        # transposing to ensure that the last two dimensions
        # of the original and dimension-reduced tensors match.
        # TODO: will using X.mean(axis=-1, keepdims=True)
        # allow us to avoid these transposes?
        X = X.transpose(2, 0)
        X = X - X.mean(axis=0)
        X = X.transpose(0, 2)
        X *= self.whitening_window

        # convolve the detrended data with the time-domain
        # filters constructed during initialization from
        # the background data, using groups to ensure that
        # the convolution is performed independently for
        # each interferometer channel
        X = torch.nn.functional.conv1d(
            X, self.whitening_filter, groups=2, padding="same"
        )

        # scale by sqrt(2 / sample_rate) for some inscrutable
        # signal processing reason beyond my understanding
        return X * (2 / self.sample_rate) ** 0.5

    def __iter__(self):
        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self.batches_per_epoch:
            raise StopIteration

        # create an array of all background
        X = self.sample_from_background()

        # create a target tensor, marking all the glitch data as 0.
        y = torch.zeros((self.batch_size,))

        # replace some of this data with glitches if
        # we have glitch data to use
        if self.glitch_sampler is not None:
            hanford_glitches, livingston_glitches = self.glitch_sampler.sample(
                self.num_glitches, self.kernel_size
            )

            if hanford_glitches is not None:
                X[: len(hanford_glitches), 0] = hanford_glitches
                idx = len(hanford_glitches)
            else:
                idx = 0

            if livingston_glitches is not None:
                slc = slice(idx, idx + len(livingston_glitches))
                X[slc, 1] = livingston_glitches

        # inject waveforms into the background if we have
        # generated waveforms to sample from
        if self.waveform_sampler is not None:
            waveforms = self.waveform_sampler.sample(
                self.num_waveforms, self.kernel_size
            )
            waveforms = np.stack(waveforms)
            waveforms = torch.Tensor(waveforms).to(self.device)

            X[-self.num_waveforms :] += waveforms
            y[-self.num_waveforms :] = 1

        # send targets to device
        y = torch.Tensor(y).to(self.device)

        X = self.whiten(X)
        self._batch_idx += 1
        return X, y
