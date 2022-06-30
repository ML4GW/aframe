from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries

from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.transforms.whitening import DEFAULT_FFTLENGTH
from bbhnet.data.utils import sample_kernels
from bbhnet.data.waveform_sampler import WaveformSampler


def _load_background(fname: str):
    # TODO: maybe these are gwf and we resample?
    with h5py.File(fname, "r") as f:
        background = f["hoft"][:]

        # grab the timestamps from the dataset for geocent sampling
        t0 = f["t0"][()]
    return background, t0


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
        trigger_distance: float = 0,
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
            trigger_distance:
                The maximum number of seconds the t0 of signals and glitches
                can lie away from the edge of sampled kernels
        """

        # default behavior is set trigger_distance to half
        # kernel length
        # with this setting t0 can lie anywhere in the kernel
        self.trigger_distance = trigger_distance

        self.trigger_distance_size = self.trigger_distance * sample_rate

        # sanity check our fractions
        assert 0 <= waveform_frac <= 1
        assert 0 <= glitch_frac <= 1

        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        # load in the background data
        hanford_background, t0 = _load_background(hanford_background)
        livingston_background, _ = _load_background(livingston_background)
        assert len(hanford_background) == len(livingston_background)

        self.hanford_background = torch.tensor(
            hanford_background, dtype=torch.float64
        )
        self.livingston_background = torch.tensor(
            livingston_background, dtype=torch.float64
        )

        # if we specified a waveform sampler, fit its snr
        # computation to the given background asd
        if waveform_sampler is not None:
            assert waveform_frac > 0
            self.fit_waveform_sampler(waveform_sampler, t0)

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
                glitch_sampler = GlitchSampler(glitch_sampler)
            self.glitch_sampler = glitch_sampler
        else:
            # likewise, ensure that we didn't indicate that
            # we expected any glitches in the batch
            assert glitch_frac == 0
            self.num_glitches = 0
            self.glitch_sampler = None

        # initialize our device to be cpu so that we
        # have to be explicit about forcing the background
        # tensors to the device at lower precision
        self.device = "cpu"

        # make sure that we have at least _some_
        # pure background in each batch
        assert (self.num_waveforms + self.num_glitches) < batch_size

    def fit_waveform_sampler(
        self, waveform_sampler: WaveformSampler, t0: float
    ) -> None:
        backgrounds = OrderedDict(
            H1=self.hanford_background, L1=self.livingston_background
        )
        asds = []
        for channel, background in backgrounds.items():
            background = background.cpu().numpy()
            ts = TimeSeries(background, dt=1 / self.sample_rate)
            asd = ts.asd(
                fftlength=DEFAULT_FFTLENGTH, window="hanning", method="median"
            )
            asd.channel = channel + ":STRAIN"
            asds.append(asd)

        tf = t0 + len(background) / self.sample_rate
        waveform_sampler.fit(t0, tf, *asds)

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
            self.hanford_background,
            self.kernel_size,
            self.trigger_distance_size,
            self.batch_size,
        )
        livingston_kernels = sample_kernels(
            self.livingston_background,
            self.kernel_size,
            self.trigger_distance_size,
            self.batch_size,
        )

        # interweave these kernels along the 0th axis so that
        # a reshape puts them in the right channel dimension
        kernels = zip(hanford_kernels, livingston_kernels)
        kernels = [i for j in kernels for i in j]
        kernels = torch.stack(kernels, dim=0)
        return kernels.reshape(self.batch_size, 2, -1)

    def to(self, device: str):
        """
        Map the background tensors to the indicated device
        and downcast to float32. Implement the downcasting
        here because the assumption is that if you're moving
        to the device, you're ready for training, and you
        (in general) don't want to train at double precision
        """
        self.hanford_background = self.hanford_background.to(device).type(
            torch.float32
        )
        self.livingston_background = self.livingston_background.to(
            device
        ).type(torch.float32)

        if self.glitch_sampler is not None:
            self.glitch_sampler.to(device)

        # store the indicated device so that we know where to
        # move our simulated waveforms and target tensors
        # at data loading time
        self.device = device

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
                self.num_glitches, self.kernel_size, self.trigger_distance_size
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
                self.num_waveforms,
                self.kernel_size,
                self.trigger_distance_size,
            )
            waveforms = np.stack(waveforms)
            waveforms = torch.Tensor(waveforms).to(self.device)

            X[-self.num_waveforms :] += waveforms
            y[-self.num_waveforms :] = 1

        # send targets to device
        y = torch.Tensor(y).to(self.device)

        self._batch_idx += 1
        return X, y
