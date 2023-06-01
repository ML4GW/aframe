import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torchmetrics.classification import BinaryAUROC

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
    tracker: LocalTracker
    background: torch.Tensor
    waveforms: torch.Tensor
    sample_rate: float
    stride: float
    injection_stride: float
    kernel_length: float
    batch_size: int
    pool_length: float
    integration_length: float
    livetime: float
    shift: float
    max_fpr: float
    device: str

    def __post_init__(self):
        print("here!")
        self.auroc = BinaryAUROC(max_fpr=self.max_fpr)
        self.duration = self.background.shape[-1] / self.sample_rate
        self.kernel_size = int(self.kernel_length * self.sample_rate)
        self.stride_size = int(self.stride * self.sample_rate)
        self.pool_size = int(self.pool_length / self.stride)

        integration_size = int(self.integration_length / self.stride)
        self.window = torch.ones((1, 1, integration_size)) / integration_size
        self.window = self.window.to(self.device)
        self.integration_size = integration_size

        self._injection_idx = 0
        self._injection_step = int(self.injection_stride // self.stride)

    def steps_for_shift(self, shift: float):
        shift = abs(shift)  # doesn't matter which direction
        return (self.duration - shift - self.kernel_length) // self.stride + 1

    def shift_background(self, shift: float):
        if not shift:
            return self.background

        # TODO: how to generalize for >1 IFO?
        shift_size = int(shift * self.sample_rate)
        if shift_size > 0:
            return np.stack(
                [
                    self.background[0, :-shift_size],
                    self.background[1, shift_size:],
                ]
            )
        else:
            return np.stack(
                [
                    self.background[0, -shift_size:],
                    self.background[1, :shift_size],
                ]
            )

    def iter_shift(self, shift):
        num_steps = self.steps_for_shift(shift)
        print(shift, num_steps)
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
        preds = preds.view(1, 1, -1)
        preds = torch.nn.functional.pad(preds, [self.integration_size - 1, 0])
        preds = torch.nn.functional.conv1d(preds, self.window)
        preds = torch.nn.functional.max_pool1d(
            preds, self.pool_size, stride=self.pool_size
        )
        return preds[0, 0]

    def inject(self, X: torch.Tensor) -> torch.Tensor:
        X = X[:: self._injection_step]
        start = self._injection_idx
        stop = start + len(X)
        waveforms = self.waveforms[start:stop]

        start = waveforms.shape[-1] // 2 - self.kernel_size // 2
        stop = start + self.kernel_size
        waveforms = waveforms[:, :, int(start) : int(stop)]
        waveforms = waveforms.to(X.device)
        return X[: len(waveforms)] + waveforms

    @torch.no_grad()
    def infer_shift(self, model: torch.nn.Module, shift: float):
        preds, inj_preds = [], []
        for X in self.iter_shift(shift):
            y = model(X)[:, 0]
            preds.append(y)

            if self._injection_idx >= len(self.waveforms):
                continue

            X = self.inject(X)
            y = model(X)[:, 0]
            inj_preds.append(y)
            self._injection_idx += len(X)

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
        while T < self.livetime:
            shift = i * self.shift
            preds, inj_preds = self.infer_shift(model, shift)
            predictions.append(preds)
            if inj_preds is not None:
                inj_predictions.append(inj_preds)

            # do the positive and negative shifts
            # for each shift value
            T += self.duration - abs(shift)
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
