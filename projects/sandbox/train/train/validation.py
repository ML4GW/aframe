import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import h5py
import numpy as np
import torch
from torchmetrics.classification import BinaryAUROC

import ml4gw.gw as gw
from ml4gw.utils.slicing import unfold_windows


@dataclass
class LocalTracker:
    """Callable which handles metric evaluation and model checkpointing

    Given a model, its most recent train loss measurement,
    and tensors of predictions on background, glitch, and
    signal datasets, this evaluates a series of `Metrics`
    and records them alongside the training loss in a
    dictionary for recording training progress. The weights
    of the best performing model according to the monitored
    metric will be saved in `logdir` as `weights.pt`. Will also
    optionally check for early stopping and perform periodic
    checkpointing of the model weights.

    Args:
        logdir:
            Directory to save artifacts to, including best-performing
            model weights, training history, and an optional subdirectory
            for periodic checkpointing.
        monitor:
            Metric which will be used for deciding which model
            weights are "best-performing." Must be one of the
            keyword arguments passed to `LocalTracker.log`
        early_stop:
            Number of epochs to go without an improvement in
            the monitored metric at the given threshold before
            training should be terminated. If left as `None`,
            never prematurely terminate training.
        checkpoint_every:
            Indicates a frequency at which to checkpoint model
            weights in terms of number of epochs. If left as
            `None`, model weights won't be checkpointed and
            only the weights which produced the best score
            on the monitored metric at the indicated threshold
            will be saved.
    """

    logdir: Path
    monitor: str
    early_stop: Optional[int] = None
    checkpoint_every: Optional[int] = None

    def __post_init__(self):
        self.history = defaultdict(list)
        if self.checkpoint_every is not None:
            self._checkpoint_dir = self.logdir / "checkpoints"
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best = -1  # best monitored metric value so far
        self.step = 0  # epoch counter
        self._since_last = 0  # epochs since last best monitored metric

    def log(self, model: torch.nn.Module, metrics: Dict[str, float]) -> bool:
        """
        Log performance metrics for a particular iteration
        of the model. Returns a flag indicating whether
        an early stopping condition has been met and therefore
        training ought to terminate.
        """
        if self.monitor not in metrics:
            raise ValueError(
                "No metric {} available to track in metrics {}".format(
                    self.monitor, ", ".join(metrics)
                )
            )

        logs = []
        for metric, value in metrics.items():
            self.history[metric].append(value)
            logs.append(f"{metric}: {value:0.3e}")
        logging.info(", ".join(logs))

        self.step += 1
        with h5py.File(self.logdir / "history.h5", "w") as f:
            for key, value in self.history.items():
                f[key] = value

        if (
            self.checkpoint_every is not None
            and not self.step % self.checkpoint_every
        ):
            epoch = str(self.step).zfill(4)
            fname = self._checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), fname)

        if metrics[self.monitor] > self.best:
            logging.info(
                f"{self.monitor} achieved new best score, saving weights"
            )
            fname = self.logdir / "weights.pt"
            torch.save(model.state_dict(), fname)
            self._since_last = 0
            self.best = metrics[self.monitor]
        elif self.early_stop is not None:
            self._since_last += 1
            if self._since_last >= self.early_stop:
                return True
        return False


@dataclass
class Validator:
    """
    Callable class that takes as input a model and training loss
    value at each epoch and returns a flag indicating whether to
    terminate training or not. This is decided by the current
    model's performance on the validation dataset. After each
    epoch, the background validation data is time-shifted to
    create `livetime` seconds worth of background.

    A foreground dataset is created by injecting validation
    waveforms into the background. Each waveform is added onto the
    background `num_views` times, with each instance containing a
    different, overlapping, `kernel_length`-second portion of the
    signal.

    The model is evaluated on the background and foreground and
    the output is integrated and pooled to identify events. From
    this, the area under the ROC curve up to a false positive rate
    of `max_fpr` is calculated as the validation metric. This value
    is used by `tracker` to decide whether training should be
    stopped.

    Args:
        tracker:
            `LocalTracker` object which keeps track of model
            performance on different metrics, creates model
            checkpoints, and determines if training should be
            stopped
        background:
            Tensor containing background data for each interferometer.
            Expected to be of shape
            `(num_segments, num_channels, num_samples)`.
        waveforms:
            Tensor containing injections for each interferometer
        psd_estimator:
            Callable that takes a timeseries and returns a PSD
            and a timeseries. Using the `PsdEstimator` in
            `aframe.train.data_structures`, this will return
            the PSD of an intial segment of the given timeseries
            as well as the part of the timeseries not used for
            PSD calculation.
        whitener:
            Callable that takes a timeseries and a PSD and returns the
            whitened timeseries
        sample_rate:
            The rate at which the background data and waveforms have
            been sampled, specified in Hz
        stride:
            Length in seconds of the time between background kernels
        injection_stride:
            Length in seconds of the time between injections
        snr_thresh:
            Minimum allowable SNR of an injection. Injections with SNRs
            below this value will be rescaled to have an SNR of this
            value.
        highpass:
            Minimum frequency over which to compute SNR values
            for waveform injection, in Hz. If left as `None`, the
            SNR will be computed over all frequency bins.
        kernel_length:
            The length, in seconds, of each batch element. This
            does not include the length of data removed after
            whitening.
        batch_size:
            Number of kernels to perform inference over at once
        pool_length:
            Length in seconds over which to perform max pooling of
            network output after integration
        integration_length:
            Length in seconds of the boxcar window with which to
            convolve network output
        livetime:
            Length in seconds of background to create via time shifts
        shift:
            Length in seconds of the shift step size. During time
            shifting, the timeseries of the nth interferometer after
            the first will be shifted by `n * shift` seconds in both
            directions, and then `2 * n * shift` seconds, etc.,
            until `livetime` seconds of background have been created.
        max_fpr:
            The value of the false positive rate up to which the area
            under the ROC curve will be calculated
        device:
            Device on which training is being performed
        num_views:
            Number of instances to create of each injection. Each
            instance will contain a different, possibly overlapping,
            `kernel_length`-second portion of the injection. The
            coalescence point of the the signal, assumed to be in
            the center of the timeseries, will be contained in each
            instance.
        pad:
            Amount of time in seconds on either side of an injection
            that will not be included in any of the `num_views`
            instances
    """

    tracker: LocalTracker
    background: torch.Tensor
    waveforms: torch.Tensor
    psd_estimator: Callable
    whitener: Callable
    sample_rate: float
    stride: float
    injection_stride: float
    snr_thresh: float
    highpass: float
    kernel_length: float
    batch_size: int
    pool_length: float
    integration_length: float
    livetime: float
    shift: float
    max_fpr: float
    device: str
    num_views: int = 5
    pad: float = 0.1

    def __post_init__(self):
        self.auroc = BinaryAUROC(max_fpr=self.max_fpr)
        self.num_segments = self.background.shape[0]
        self.durations = [
            background.shape[-1] / self.sample_rate
            for background in self.background
        ]
        self.num_channels = self.background.shape[1]
        self.kernel_size = int(self.kernel_length * self.sample_rate)
        self.stride_size = int(self.stride * self.sample_rate)
        self.pool_size = int(self.pool_length / self.stride)
        self.pad_size = int(self.pad * self.sample_rate)

        integration_size = int(self.integration_length / self.stride)
        self.window = torch.ones((1, 1, integration_size)) / integration_size
        self.window = self.window.to(self.device)
        self.integration_size = integration_size

        self._injection_idx = 0
        self._injection_step = int(self.injection_stride // self.stride)

    def steps_for_shift(self, shift: float):
        """Compute the number of kernels that will be taken per shift"""
        shift = abs(shift)  # doesn't matter which direction
        max_shift = shift * (self.num_channels - 1)
        return (
            self.current_duration - max_shift - self.kernel_length
        ) // self.stride + 1

    def shift_background(self, shift: float):
        """
        Return the background with the nth interferometer after
        the first being shifted by `n * shift` seconds
        """
        if not shift:
            return self.current_segment

        shift_size = int(abs(shift) * self.sample_rate)
        max_shift_size = shift_size * (self.num_channels - 1)
        remainder_size = self.current_segment.shape[-1] - max_shift_size
        start_idxs = [i * shift_size for i in range(self.num_channels)]
        if shift < 0:
            start_idxs.reverse()

        return np.stack(
            [
                ifo_background[start : start + remainder_size]
                for start, ifo_background in zip(
                    start_idxs, self.current_segment
                )
            ]
        )

    def iter_shift(self, shift):
        """
        Compute a particular shift of the background and yield
        batches of `batch_size` kernels
        """
        num_steps = self.steps_for_shift(shift)
        num_batches = (num_steps - 1) // self.batch_size + 1
        background = self.shift_background(shift)

        step_size = self.stride_size * (self.batch_size - 1) + self.kernel_size
        for i in range(int(num_batches)):
            start = i * self.batch_size * self.stride_size
            X = background[:, start : start + step_size]
            X = torch.Tensor(X).to(self.device)
            X = unfold_windows(X, self.kernel_size, stride=self.stride_size)
            yield X

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Integrate and pool the network outputs"""
        preds = preds.view(1, 1, -1)
        preds = torch.nn.functional.pad(preds, [self.integration_size - 1, 0])
        preds = torch.nn.functional.conv1d(preds, self.window)
        preds = torch.nn.functional.max_pool1d(
            preds, self.pool_size, stride=self.pool_size
        )
        return preds[0, 0]

    def threshold_snrs(self, waveforms: torch.Tensor, psds: torch.Tensor):
        """
        Compute the SNRs of a set of injections, and if any fall below
        the threshold, scale those injections to have an SNR of
        `snr_thresh`.
        """
        num_freqs = waveforms.shape[-1] // 2 + 1
        if num_freqs != psds.shape[-1]:
            psds = torch.nn.functional.interpolate(psds, (num_freqs,))

        mask = torch.linspace(0, self.sample_rate / 2, num_freqs)
        mask = mask >= self.highpass
        mask = mask.to(waveforms.device)

        snrs = gw.compute_network_snr(waveforms, psds, self.sample_rate, mask)
        target_snrs = snrs.clamp(self.snr_thresh, 1000)
        weights = target_snrs / snrs
        return waveforms * weights.view(-1, 1, 1)

    def inject(self, X: torch.Tensor, psds: torch.Tensor) -> torch.Tensor:
        # downsample the background batch to give us
        # the desired stride between injections
        X = X[:: self._injection_step]
        psds = psds[:: self._injection_step]

        # grab the next batch of injections and possibly
        # reduce the background batch size to match it
        # if we're at the end. Increment our injection
        # index counter
        start = self._injection_idx
        stop = start + len(X)
        waveforms = self.waveforms[start:stop].to(X.device)
        psds = psds[: len(waveforms)]
        X = X[: len(waveforms)]
        self._injection_idx += len(waveforms)

        # threshold the SNRs of the injections to the desired value.
        if self.snr_thresh > 0:
            waveforms = self.threshold_snrs(waveforms, psds)

        # create `num_view` instances of the injection on top of
        # the background, each showing a different, overlapping
        # portion of the signal
        kernel_size = X.shape[-1]
        center = waveforms.shape[-1] // 2
        step = (kernel_size - 2 * self.pad_size) / (self.num_views - 1)
        batch_X, batch_psd = [], []
        for i in range(self.num_views):
            start = center - self.pad_size - int(i * step)
            stop = start + kernel_size
            injected = X + waveforms[:, :, int(start) : int(stop)]

            batch_X.append(injected)
            batch_psd.append(psds)

        batch_X = torch.cat(batch_X, dim=0)
        batch_psd = torch.cat(batch_psd, dim=0)
        return batch_X, batch_psd

    def predict(self, model, X, psd):
        """Whiten the given data and pass it through the model"""
        X = self.whitener(X, psd)
        return model(X)[:, 0]

    @torch.no_grad()
    def infer_shift(self, model: torch.nn.Module, shift: float):
        """
        Evaluate the model against a particular time shift,
        integrate and pool the results, and return the model
        predictions of the background and foreground
        """
        preds, inj_preds = [], []
        for X in self.iter_shift(shift):
            X, psd = self.psd_estimator(X)
            y = self.predict(model, X, psd)
            preds.append(y)

            if self._injection_idx >= len(self.waveforms):
                continue

            X, psd = self.inject(X, psd)
            y = self.predict(model, X, psd)
            y = y.reshape(self.num_views, -1).mean(0)
            inj_preds.append(y)

        preds = torch.cat(preds)
        if inj_preds:
            inj_preds = torch.cat(inj_preds)
        else:
            inj_preds = None

        preds = self.postprocess(preds)
        return preds, inj_preds

    def __call__(self, model: torch.nn.Module, train_loss: float) -> bool:
        model.eval()
        self._injection_idx = 0
        predictions, inj_predictions = [], []
        T = 0
        i = 1
        seg_it = itertools.cycle(range(self.num_segments))
        while T < self.livetime:
            j = next(seg_it)
            self.current_segment, self.current_duration = (
                self.background[j],
                self.durations[j],
            )

            shift = i * self.shift
            preds, inj_preds = self.infer_shift(model, shift)
            predictions.append(preds)
            if inj_preds is not None:
                inj_predictions.append(inj_preds)

            # do the positive and negative shifts
            # for each shift value
            max_shift = abs(shift) * (self.num_channels - 1)
            T += self.current_duration - max_shift

            # increment the shift counter
            # once we've analyzed all the segments
            # at this current shift
            if j == (self.num_segments - 1):
                i *= -1
                if i > 0:
                    i += 1

        predictions = torch.cat(predictions)
        inj_predictions = torch.cat(inj_predictions)

        predictions = torch.cat([predictions, inj_predictions])
        targets = torch.zeros_like(predictions)
        targets[-len(inj_predictions) :] = 1

        # shuffle the prediction and target arrays up
        # front so that constant-output models don't
        # accidently come out perfect
        idx = torch.randperm(len(predictions))
        predictions = predictions[idx]
        targets = targets[idx]
        auroc = self.auroc(predictions, targets).item()

        metrics = {
            "train_loss": train_loss,
            f"valid_auroc@{self.max_fpr:0.1e}": auroc,
        }
        return self.tracker.log(model, metrics)
