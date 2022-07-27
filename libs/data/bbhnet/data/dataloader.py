from enum import Enum
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch

from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.utils import sample_kernels
from bbhnet.data.waveform_sampler import WaveformSampler


def _load_background(fname: str, frac: Optional[float] = None) -> torch.Tensor:
    # TODO: maybe these are gwf and we resample?
    with h5py.File(fname, "r") as f:
        background = f["hoft"][:]

    if frac is not None:
        N = int(frac * len(background))
        if frac < 0:
            background = background[N:]
        else:
            background = background[:N]
    return torch.tensor(background, dtype=torch.float64)


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
        frac: Optional[float] = None,
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
        # kernel length. With this setting t0 can lie anywhere
        # in the kernel
        self.trigger_distance_size = trigger_distance * sample_rate

        # sanity check our fractions
        assert 0 <= waveform_frac <= 1
        assert 0 <= glitch_frac <= 1

        # initialize our device to be cpu so that we
        # have to be explicit about forcing the background
        # tensors to the device at lower precision
        self.device = "cpu"
        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        # load in the background data
        self.hanford_background = _load_background(hanford_background, frac)
        self.livingston_background = _load_background(
            livingston_background, frac
        )
        assert len(self.hanford_background) == len(self.livingston_background)

        if waveform_sampler is not None:
            assert waveform_frac > 0

            # if the waveform sampler hasn't already been fit to
            # any data, fit it to this background for the snr
            # reweighting computation
            if waveform_sampler.background_asd is None:
                waveform_sampler.fit(
                    H1=self.hanford_background, L1=self.livingston_background
                )
            self.waveform_sampler = waveform_sampler
            self.num_waveforms = max(1, int(waveform_frac * batch_size))
        else:
            # ensure that we didn't indicate that
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
                glitch_sampler = GlitchSampler(glitch_sampler, frac=frac)
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


class Status(Enum):
    """Utility enum for keeping track of which type of data to load"""

    BACKGROUND = 0
    GLITCH = 1
    SIGNAL = 2


class DeterministicWaveformDataset:
    def __init__(
        self,
        hanford_background: str,
        livingston_background: str,
        kernel_length: float,
        stride: float,
        sample_rate: float,
        batch_size: int,
        waveform_sampler: Optional[WaveformSampler] = None,
        glitch_sampler: Union[GlitchSampler, str, None] = None,
        offset: float = 0,
        frac: Optional[float] = None,
    ) -> None:
        # initialize our device to be cpu so that we
        # have to be explicit about forcing the background
        # tensors to the device at lower precision
        self.device = "cpu"
        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.stride_size = int(stride * sample_rate)
        self.batch_size = batch_size

        # load in the background data
        hanford_background = _load_background(hanford_background, frac)
        livingston_background = _load_background(livingston_background, frac)
        assert len(hanford_background) == len(livingston_background)

        offset = offset * sample_rate
        self.waveforms = self.glitches = None
        if waveform_sampler is not None:
            if waveform_sampler.background_asd is None:
                waveform_sampler.fit(
                    H1=hanford_background, L1=livingston_background
                )

            # sample waveforms up front
            waveforms = waveform_sampler.sample(-1, self.kernel_size, offset)
            self.waveforms = torch.Tensor(waveforms)

        # load in any glitches if we specified them
        if glitch_sampler is not None:
            if isinstance(glitch_sampler, (str, Path)):
                glitch_sampler = GlitchSampler(
                    glitch_sampler, deterministic=True, frac=frac
                )
            glitches = glitch_sampler.sample(-1, self.kernel_size, offset)
            glitches = np.stack(glitches).transpose(1, 0, 2)
            self.glitches = torch.Tensor(glitches)

        # initialize some attributes for iteration
        self.background = torch.stack(
            [hanford_background, livingston_background]
        )
        self._idx = self._status = self._secondary_idx = None

    def to(self, device: str):
        """
        Map the background tensors to the indicated device
        and downcast to float32. Implement the downcasting
        here because the assumption is that if you're moving
        to the device, you're ready for training, and you
        (in general) don't want to train at double precision
        """
        self.background = self.background.to(device).type(torch.float32)
        if self.waveforms is not None:
            self.waveforms = self.waveforms.to(device)
        if self.glitches is not None:
            self.glitches = self.glitches.to(device)
        self.device = device

    def __iter__(self):
        self._idx = 0
        self._status = Status.BACKGROUND
        return self

    def __next__(self):
        cutoff = self.background.shape[-1] - self.kernel_size
        if self._status is Status.GLITCH:
            # since we always insert one glitch for each
            # IFO, we need at least 2 kernels
            cutoff -= self.stride_size

        if self._idx >= cutoff:
            # we've exhausted our current run of the background
            # dataset, so restart the index
            self._idx = 0

            # if we were iterating through background,
            # move on to the glitches
            if self._status is Status.BACKGROUND:
                if self.glitches is None and self.waveforms is None:
                    self._status = self._idx = None
                    raise StopIteration
                elif self.glitches is None:
                    self._status = Status.SIGNAL
                else:
                    self._status = Status.GLITCH

                self._secondary_idx = 0

        # slice out the next stretch of background data
        length = (self.batch_size - 1) * self.stride_size + self.kernel_size
        stop = self._idx + length
        if stop > self.background.shape[-1]:
            # if we won't be able to grab an entire batch
            # worth of data, make sure that we grab the
            # right amount to be able to evenly unroll
            stop = self.background.shape[-1]
            step_length = stop - self.kernel_size - self._idx
            num_kernels, leftover = divmod(step_length, self.stride_size)
            num_kernels += 1
            stop -= leftover
        else:
            num_kernels = self.batch_size

        # give dummy batch and extra spatial dimension
        # since unfold expects a 4D tensor (multichannel image)
        X = self.background[None, :, None, self._idx : stop]

        # TODO: give the option of unrolling X upfront?
        # Possibly even creating injections and insertions upfront too?
        unfold = torch.nn.Unfold(
            kernel_size=(1, num_kernels), dilation=(1, self.stride_size)
        )
        X = unfold(X)
        X = X.reshape(2, num_kernels, -1).transpose(1, 0)
        y = torch.zeros((len(X),))

        self._idx += num_kernels * self.stride_size
        if self._status is Status.GLITCH:
            if self._secondary_idx >= len(self.glitches):
                if self.waveforms is None:
                    self._idx = self._secondary_idx = self._status = None
                    raise StopIteration
                else:
                    self._status = Status.SIGNAL
                    self._secondary_idx = 0
            else:
                # grab at most half the batch size, since
                # we'll insert glitches into the ifos independently
                stop = self._secondary_idx + num_kernels // 2
                glitches = self.glitches[self._secondary_idx : stop]

                # slough off any excess background that won't have
                # glitches inserted into it
                X = X[: len(glitches) * 2]

                # insert the glitches into each ifo channel
                X[: len(glitches), 0] = glitches[:, 0]
                X[len(glitches) :, 1] = glitches[:, 1]

                # update our index and return the inserted tensor
                self._secondary_idx = stop

        # check this in a separate if clause in case we
        # exhausted our glitches in the one above
        if self._status is Status.SIGNAL:
            if self._secondary_idx >= len(self.waveforms):
                # we've exhausted all our waveforms, so we're done here
                self._idx = self._status = self._secondary_idx = None
                raise StopIteration

            stop = self._secondary_idx + num_kernels
            waveforms = self.waveforms[self._secondary_idx : stop]
            X = X[: waveforms.shape[0]]
            X += waveforms
            y += 1
            self._secondary_idx = stop

        # crop y in case we had to crop X to match the
        # number of waveforms or glitches
        return X, y[: len(X)].to(self.device)
