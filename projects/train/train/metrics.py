import itertools
from collections import defaultdict
from typing import Optional

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC

from utils import x_per_y


class TimeSlideAUROC(Metric):
    """
    Metric that computes the area under the ROC curve (AUROC)
    using timeslides. AUROC is a less than ideal metric
    for a lot of reasons, but it's more stable than just
    computing the recall at a fixed FAR since it effectively
    averages your recall over many thresholds. To compensate
    for the fact that we don't care about our performance at
    higher FPRs, we only measure the AUROC up to a fixed
    false alarm rate.

    Simulates test-time scenario by pooling background
    events from within a given timeslide. For this reason,
    metric is iteratively updated with both foreground
    and background predictions, as well as an index indicating
    the shift the predictions were made on.

    Torch metrics handles aggregating the predictions from
    multiple workers on the backend, so this is automatically
    compatible with distributed evaluation. The only trick is
    making sure that your timeslides get split up among your
    devices.
    """

    def __init__(
        self, max_fpr: float, stride: float, pool_length: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = BinaryAUROC(max_fpr)
        pool_size = int(pool_length / stride)
        pool_stride = int(pool_size // 2)
        self.pool = torch.nn.MaxPool1d(pool_size, pool_stride, ceil_mode=True)

        self.add_state("shifts", default=[])
        self.add_state("background", default=[])
        self.add_state("foreground", default=[])

    def update(
        self, shift: int, background: torch.Tensor, foreground: torch.Tensor
    ) -> None:
        self.shifts.append(torch.Tensor([shift]).to(background.device))
        self.background.append(background)
        self.foreground.append(foreground)

    def compute(self):
        foreground, background = [], defaultdict(list)
        for i, bg, fg in zip(self.shifts, self.background, self.foreground):
            foreground.append(fg)
            background[i.item()].append(bg)
        foreground = torch.cat(foreground)

        pooled_background = []
        for bg in background.values():
            bg = torch.cat(bg).view(1, 1, -1)
            bg = self.pool(bg).view(-1)
            pooled_background.append(bg)
        background = torch.cat(pooled_background)

        # concatenate these with view-averaged foreground
        # predictions to constitute our predicted outputs
        y_pred = torch.cat([background, foreground])

        # now create ground-truth labels
        y = torch.zeros_like(y_pred)
        y[len(background) :] = 1

        # shuffle the prediction and target arrays up
        # front so that constant-output models don't
        # accidently come out perfect
        idx = torch.randperm(len(y_pred))
        y_pred = y_pred[idx]
        y = y[idx]
        return self.metric(y_pred, y)


class TimeSlide(torch.utils.data.IterableDataset):
    """
    Iterate through a timeseries in `stride_size * batch_size`
    chunks along the time dimension to produce batches of
    `kernel_size` tensors after unfolding.

    Args:
        timeseries: 2D timeseries data to iterate through
        shift_size:
            Number of samples by which to shift interferometers
            with respect to one another. If `shift_size`
            is positive, the 0th channel with be unshifted and each
            channel after it will be shifted left by another factor
            of `shift_size` with respect to the one before it (i.e.
            the first sample returned will be the `shift_size * i`th
            sample in the un-shifted channel, where `i` represents
            the channel index). If `shift_size` is negative, the
            last channel will be unshifted and shifting will be
            applied in reverse from it.
        kernel_size:
            The intended size along the time dimension of the
            unfolded batches produced by this timeseries.
        stride_size:
            The number of samples between the windows in the
            unfolded batches.
        batch_size:
            The maximum number of windows in each unfolded batch.
            The last batch may be smaller than this.
        start:
            Index along the timeseries to begin generating data
        stop:
            Last index along the timeseries from which to generate data
    """

    def __init__(
        self,
        timeseries: torch.Tensor,
        shift_size: int,
        kernel_size: float,
        stride_size: float,
        batch_size: int,
        start: int = 0,
        stop: int = -1,
    ) -> None:
        self.timeseries = timeseries
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.batch_size = batch_size
        self.shift_size = shift_size
        self.start = start
        self.stop = stop if stop != -1 else timeseries.size(-1)
        self.step_size = stride_size * (batch_size - 1) + kernel_size

        shifts = [i * abs(shift_size) for i in range(len(timeseries))]
        if shift_size < 0:
            shifts.reverse()
        self.shifts = shifts

    @property
    def max_shift(self):
        num_channels = len(self.timeseries)
        return abs(self.shift_size) * (num_channels - 1)

    @property
    def num_steps(self):
        size = self.stop - self.start - self.max_shift - self.kernel_size
        return size // self.stride_size + 1

    def new_bounds(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> "TimeSlide":
        """
        Basically return a new timeslide that points to the
        same timeseries but uses different start and stop
        points. Useful for breaking a timeslide up between
        devices for distributed evaluation.
        """
        return TimeSlide(
            self.timeseries,
            self.shift_size,
            self.kernel_size,
            self.stride_size,
            self.batch_size,
            self.start if start is None else start,
            stop or self.stop,
        )

    def __len__(self) -> int:
        return x_per_y(self.num_steps, self.batch_size)

    def __iter__(self):
        for i in range(len(self)):
            start = self.start + i * self.batch_size * self.stride_size
            X = []
            for j, offset in enumerate(self.shifts):
                offset = start + offset
                stop = min(self.stop, offset + self.step_size)
                x = self.timeseries[j, offset:stop]

                # if we don't have enough for a full batch,
                # make sure that we cut off x to ensure that
                # we can evenly produce a smaller batch
                if len(x) < self.step_size:
                    size = len(x) - self.kernel_size
                    strides = int(size // self.stride_size)
                    stop = strides * self.stride_size + self.kernel_size
                    x = x[:stop]
                X.append(x)

            # if any of the shifted arrays are too short at
            # the ends of the timeseries, make sure all the
            # arrays we try to stack are the same size
            minlen = min([len(x) for x in X])
            X = [x[:minlen] for x in X]
            yield torch.stack(X)


def get_timeslides(
    timeseries: torch.Tensor,
    livetime: float,
    sample_rate: float,
    sample_length: float,
    stride: float,
    batch_size: int,
):
    """
    Generate timeslides of the provided `timeseries` until
    `livetime` they've produced `livetime` seconds of background.
    For distributed evaluation, timelides are broken up to ensure
    each device performs and equal amount of inference.

    `timeseries` is a (n_val_segments, n_channels, n_samples) tensor
    """

    kernel_size = int(sample_length * sample_rate)
    stride_size = int(stride * sample_rate)

    # create alternating positive and negative
    # shifts until we have enough livetime
    livetime = livetime + 0  # so our inplace ops don't have unintended effects
    shifts = zip(itertools.count(1, 1), itertools.count(-1, -1))
    shifts = itertools.chain.from_iterable(shifts)
    shifts = itertools.takewhile(lambda _: livetime > 0, shifts)

    timeslides = []
    for shift in shifts:
        for ts in timeseries:
            timeslide = TimeSlide(
                ts,
                int(shift * sample_rate),
                kernel_size,
                stride_size,
                batch_size,
            )
            duration = ts.size(-1) / sample_rate
            duration -= sample_length

            dur = duration - timeslide.max_shift / sample_rate
            livetime -= dur
            timeslides.append(timeslide)
            if livetime <= 0:
                break

    # chop off any excess time from the last timeslide
    if livetime < 0:
        remainder = dur + livetime + sample_length
        stop = int(remainder * sample_rate)
        timeslides[-1].stop = stop

    # check if we're running distributed, and if not
    # run all the timeslides on the current device
    try:
        world_size = torch.distributed.get_world_size()
    except ValueError:
        return timeslides, sum([len(ts) for ts in timeslides])
    if world_size == 1:
        return timeslides, sum([len(ts) for ts in timeslides])

    # if we're running with more than one device,
    # break up the timeslides such that we do
    # roughly an equal number of steps per device
    total_steps = sum([i.num_steps for i in timeslides])
    steps_per_dev = x_per_y(total_steps, world_size)

    # for each timeslide, check if adding the full
    # timeslide to the current device would go over
    # the target number of steps
    timeslides_per_dev = [[]]
    it = iter(timeslides)
    timeslide = next(it)
    while True:
        current_steps = sum([t.num_steps for t in timeslides_per_dev[-1]])
        if current_steps + timeslide.num_steps > steps_per_dev:
            # stop the current timeslide at the index
            # that gives us the desired number of steps
            num_steps = steps_per_dev - current_steps + 1

            stop = (
                timeslide.start
                + timeslide.max_shift
                + kernel_size
                + num_steps * stride_size
            )
            new = timeslide.new_bounds(stop=stop)

            # add this truncated timeslide to our current list,
            # then create a new list to start adding to
            timeslides_per_dev[-1].append(new)
            timeslides_per_dev.append([])

            # start a new truncated timeslide one stride
            # after the last step of the previous one.
            # Include the padding we'll need, TODO:
            # should this be + stride_size - kernel_size?
            start = stop - stride_size - kernel_size
            timeslide = timeslide.new_bounds(start=start)
        elif current_steps + timeslide.num_steps == steps_per_dev:
            break
        else:
            # if this timeslide won't put us over, add the
            # whole thing as is and try to move on to the next
            timeslides_per_dev[-1].append(timeslide)
            try:
                timeslide = next(it)
            except StopIteration:
                break

    # TODO: fix me: this is a hack to get around
    # the fact that the timeslides in total don't
    # combine to the same number of batches, leading
    # to hanging when validating with distributed training;
    # we should probably just move to a simpler method
    # for distributing the timeslides
    lengths = []
    for dev in timeslides_per_dev:
        length = 0
        for ts in dev:
            length += len(ts)
        lengths.append(length)
    minimum = min(lengths)

    global_rank = torch.distributed.get_rank()
    return timeslides_per_dev[global_rank], minimum
