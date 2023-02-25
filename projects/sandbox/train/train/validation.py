import logging
import pickle
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from train.utils import split

if TYPE_CHECKING:
    from train.waveform_injection import BBHNetWaveformInjection


class Metric(torch.nn.Module):
    """
    Abstract class for representing a metric which
    needs to be evaluated at particular threshold(s).
    Inherits from `torch.nn.Module` so that parameters
    for calculating the metric of interest can be
    saved as `buffer`s and moved to appropriate devices
    via `Metric.to`. Child classes should override `call`
    for actually computing the metric values of interest.
    """

    def __init__(self, thresholds) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.values = [0.0 for _ in thresholds]

    def update(self, metrics):
        try:
            metric = metrics[self.name]
        except KeyError:
            metric = {}
            metrics[self.name] = {}

        for threshold, value in zip(self.thresholds, self.values):
            try:
                metric[threshold].append(value)
            except KeyError:
                metric[threshold] = [value]

    def call(self, backgrounds, glitches, signals):
        raise NotImplementedError

    def forward(self, backgrounds, glitches, signals):
        values = self.call(backgrounds, glitches, signals)
        self.values = [v for v in values.cpu().numpy()]
        return values

    def __str__(self):
        tab = " " * 8
        string = ""
        for threshold, value in zip(self.thresholds, self.values):
            string += f"\n{tab}{self.param} = {threshold}: {value:0.3f}"
        return self.name + " @:" + string

    def __getitem__(self, threshold):
        try:
            idx = self.thresholds.index(threshold)
        except ValueError:
            raise KeyError(str(threshold))
        return self.values[idx]

    def __contains__(self, threshold):
        return threshold in self.thresholds


class MultiThresholdAUROC(Metric):
    name = "AUROC"
    param = "max_fpr"

    def call(self, signal_preds, background_preds):
        x = torch.cat([signal_preds, background_preds])
        y = torch.zeros_like(x)
        thresholds = torch.Tensor(self.thresholds).to(y.device)
        y[: len(signal_preds)] = 1

        idx = torch.argsort(x, descending=True)
        y = y[idx]

        tpr = torch.cumsum(y, -1) / y.sum()
        fpr = torch.cumsum(1 - y, -1) / (1 - y).sum()
        dfpr = fpr.diff()
        dtpr = tpr.diff()

        mask = fpr[:-1, None] <= thresholds
        dfpr = dfpr[:, None] * mask
        integral = (tpr[:-1, None] + dtpr[:, None] * 0.5) * dfpr
        return integral.sum(0)


class BackgroundAUROC(MultiThresholdAUROC):
    def __init__(
        self, kernel_size: int, stride: int, thresholds: List[float]
    ) -> None:
        super().__init__(thresholds)
        self.kernel_size = kernel_size
        self.stride = stride

    def call(self, background, _, signal):
        background = background.unsqueeze(0)
        background = torch.nn.functional.max_pool1d(
            background, kernel_size=self.kernel_size, stride=self.stride
        )
        background = background[0]
        return super().call(signal, background)


class BackgroundRecall(Metric):
    """
    Computes the recall of injected signals (fraction
    of total injected signals recovered) at the
    detection statistic threshold given by each of the
    top `k` background "events."

    Background predictions are max-pooled along the time
    dimension using the indicated `kernel_size` and
    `stride` to keep from counting multiple events from
    the same phenomenon.

    Args:
        kernel_size:
            Size of the window, in samples, over which
            to max pool background predictions.
        stride:
            Number of samples between consecutive
            background max pool windows.
        k:
            Max number of top background events against
            whose thresholds to evaluate signal recall.
    """

    name = "recall vs. background"
    param = "k"

    def __init__(self, kernel_size: int, stride: int, k: int = 5) -> None:
        super().__init__([i + 1 for i in range(k)])
        self.kernel_size = kernel_size
        self.stride = stride
        self.k = k

    def call(self, background, _, signal):
        background = background.unsqueeze(0)
        background = torch.nn.functional.max_pool1d(
            background, kernel_size=self.kernel_size, stride=self.stride
        )
        background = background[0]
        topk = torch.topk(background, self.k).values
        recall = (signal.unsqueeze(1) >= topk).sum(0) / len(signal)
        return recall


class GlitchRecall(Metric):
    """
    Computes the recall of injected signals (fraction
    of total injected signals recovered) at the detection
    statistic threshold given by each of the glitch
    specificity values (fraction of glitches rejected)
    specified.

    Args:
        specs:
            Glitch specificity values against which to
            compute detection statistic thresholds.
            Represents the fraction of glitches that would
            be rejected at a given threshold.
    """

    name = "recall vs. glitches"
    param = "specificity"

    def __init__(self, specs: Sequence[float]) -> None:
        for i in specs:
            assert 0 <= i <= 1
        super().__init__(specs)
        self.register_buffer("specs", torch.Tensor(specs))

    def call(self, _, glitches, signal):
        qs = torch.quantile(glitches.unsqueeze(1), self.specs)
        recall = (signal.unsqueeze(1) >= qs).sum(0) / len(signal)
        return recall


@dataclass
class Recorder:
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
            weights are "best-performing"
        threshold:
            Threshold value for the monitored metric at which
            to evaluate the model's performance.
        additional:
            Any additional metrics to evaluate during training
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
    monitor: Metric
    threshold: float
    additional: Optional[Sequence[Metric]] = None
    early_stop: Optional[int] = None
    checkpoint_every: Optional[int] = None

    def __post_init__(self):
        if self.threshold not in self.monitor:
            raise ValueError(
                "Metric {} has no threshold {}".format(
                    self.monitor.name, self.threshold
                )
            )
        self.history = {"train_loss": []}

        if self.checkpoint_every is not None:
            (self.logdir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.best = -1  # best monitored metric value so far
        self._i = 0  # epoch counter
        self._since_last = 0  # epochs since last best monitored metric

    def checkpoint(
        self, model: torch.nn.Module, metrics: Dict[str, float]
    ) -> bool:
        self._i += 1
        with open(self.logdir / "history.pkl", "wb") as f:
            pickle.dump(self.history, f)

        if (
            self.checkpoint_every is not None
            and not self._i % self.checkpoint_every
        ):
            epoch = str(self._i).zfill(4)
            fname = self.logdir / "checkpoints" / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), fname)

        if self.monitor[self.threshold] > self.best:
            fname = self.logdir / "weights.pt"
            torch.save(model.state_dict(), fname)
            self._since_last = 0
            self.best = self.monitor[self.threshold]
        elif self.early_stop is not None:
            self._since_last += 1
            if self._since_last >= self.early_stop:
                return True
        return False

    def __call__(
        self,
        model: torch.nn.Module,
        train_loss: float,
        background: torch.Tensor,
        glitches: torch.Tensor,
        signal: torch.Tensor,
    ) -> bool:
        self.history["train_loss"].append(train_loss)
        self.monitor(background, glitches, signal)
        self.monitor.update(self.history)

        msg = f"Summary:\nTrain loss: {train_loss:0.3e}"
        msg += f"\nValidation {self.monitor}"
        if self.additional is not None:
            for metric in self.additional:
                metric(background, glitches, signal)
                metric.update(self.history)
                msg += f"\nValidation {metric}"
        logging.info(msg)

        return self.checkpoint(model, self.history)


def make_background(
    background: np.ndarray, kernel_size: int, stride_size: int
) -> torch.Tensor:
    num_ifos, size = background.shape
    num_kernels = (size - kernel_size) // stride_size + 1
    num_kernels = int(num_kernels)

    stop = (num_kernels - 1) * stride_size + kernel_size
    background = background[:, :stop]
    background = torch.Tensor(background)[None, :, None]

    # fold out into windows up front
    background = torch.nn.functional.unfold(
        background, (1, num_kernels), dilation=(1, stride_size)
    )

    # some reshape magic having to do with how the
    # unfold op orders things. Don't worry about it
    background = background.reshape(num_ifos, num_kernels, kernel_size)
    background = background.transpose(1, 0)
    return background


def make_glitches(
    glitches: Sequence[np.ndarray],
    background: torch.Tensor,
    glitch_frac: float,
) -> torch.Tensor:
    if len(glitches) != background.size(1):
        raise ValueError(
            "Number of glitch tensors {} doesn't match number "
            "of interferometers {}".format(len(glitches), background.size(1))
        )

    h1_glitches, l1_glitches = map(torch.Tensor, glitches)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1
    num_coinc = int(glitch_frac**2 * num_glitches / (1 + glitch_frac**2))
    if num_coinc > min(num_h1, num_l1):
        raise ValueError(
            f"There are more coincident glitches ({num_coinc}) that there "
            "are glitches in one of the ifo glitch datasets. Hanford: "
            "{num_h1}, Livingston: {num_l1}"
        )

    h1_coinc, h1_glitches = split(h1_glitches, num_coinc / num_h1, 0)
    l1_coinc, l1_glitches = split(l1_glitches, num_coinc / num_l1, 0)
    coinc = torch.stack([h1_coinc, l1_coinc], axis=1)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1 + num_coinc

    # if we need to create duplicates of some of our
    # background to make this work, figure out how many
    background = repeat(background, num_glitches)

    # now insert the glitches
    kernel_size = background.size(2)
    start = h1_glitches.shape[-1] // 2 - kernel_size // 2
    slc = slice(start, start + kernel_size)

    background[:num_h1, 0] = h1_glitches[:, slc]
    background[num_h1:-num_coinc, 1] = l1_glitches[:, slc]
    background[-num_coinc:] = coinc[:, :, slc]

    return background


def repeat(X: torch.Tensor, max_num: int) -> torch.Tensor:
    """
    Repeat a 3D tensor `X` along its 0 dimension until
    it has length `max_num`.
    """

    repeats = ceil(max_num / len(X))
    X = X.repeat(repeats, 1, 1)
    return X[:max_num]


class Validator:
    def __init__(
        self,
        recorder: Callable,
        background: np.ndarray,
        glitches: Sequence[np.ndarray],
        injector: "BBHNetWaveformInjection",
        kernel_length: float,
        stride: float,
        sample_rate: float,
        batch_size: int,
        glitch_frac: float,
        device: str,
    ) -> None:
        """Callable class for evaluating model validation scores

        Computes model outputs on background, glitch,
        and signal datasets at call time and passes them
        to a `recorder` for evaluation and checkpointing.

        Args:
            recorder:
                Callable which accepts the model being evaluated,
                most recent training loss, and the predictions on
                background, glitch, and signal data and returns a
                boolean indicating whether training should terminate
                or not.
            background:
                2D timeseries of interferometer strain data. Will be
                split into windows of length `kernel_length`, sampled
                every `stride` seconds. Glitch and signal data will be
                augmented onto this dataset
            glitches:
                Each element of `glitches` should be a 2D array
                containing glitches from each interferometer, with
                the 0th axis used to enumerate individual glitches
                and the 1st axis corresponding to time.
            injector:
                A `BBHNetWaveformInjection` object for sampling
                waveforms. Preferring this to an array of waveforms
                for the time being so that we can potentially do
                on-the-fly SNR reweighting during validation. For now,
                waveforms are sampled with no SNR reweighting.
            kernel_length:
                The length of windows to sample from the background
                in seconds.
            stride:
                The number of seconds between sampled background
                windows.
            sample_rate:
                The rate at which all relevant data arrays have
                been sampled in Hz
            batch_size:
                Number of samples over which to compute model
                predictions at call time
            glitch_frac:
                Rate at which interferometer channels are
                replaced with glitches during training. Used
                to compute the fraction of validation glitches
                which should be sampled coincidentally.
            device:
                Device on which to perform model evaluation.
        """
        self.device = device

        # move all our validation metrics to the
        # appropriate device
        recorder.monitor.to(device)
        if recorder.additional is not None:
            for metric in recorder.additional:
                metric.to(device)
        self.recorder = recorder

        kernel_size = int(kernel_length * sample_rate)
        stride_size = int(stride * sample_rate)

        # create a datset of pure background
        background = make_background(background, kernel_size, stride_size)
        self.background_loader = self.make_loader(background, batch_size)

        # now repliate that dataset but with glitches inserted
        # into either or both interferometer channels
        glitch_background = make_glitches(glitches, background, glitch_frac)
        self.glitch_loader = self.make_loader(glitch_background, batch_size)

        # 3. create a tensor of background with waveforms injected
        waveforms, _ = injector.sample(-1)
        signal_background = repeat(background, len(waveforms))

        start = waveforms.shape[-1] // 2 - kernel_size // 2
        stop = start + kernel_size
        signal_background += waveforms[:, :, start:stop]
        self.signal_loader = self.make_loader(signal_background, batch_size)

    def make_loader(self, X: torch.Tensor, batch_size: int):
        dataset = torch.utils.data.TensorDataset(X)
        return torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=batch_size,
            pin_memory_device=self.device,
        )

    def get_predictions(
        self, loader: Iterable[Tuple[torch.Tensor]], model: torch.nn.Module
    ) -> torch.Tensor:
        preds = []
        for (X,) in loader:
            X = X.to(self.device)
            y_hat = model(X)[:, 0]
            preds.append(y_hat)
        return torch.cat(preds)

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, train_loss: float) -> bool:
        background_preds = self.get_predictions(self.background_loader, model)
        glitch_preds = self.get_predictions(self.glitch_loader, model)
        signal_preds = self.get_predictions(self.signal_loader, model)

        return self.recorder(
            model, train_loss, background_preds, glitch_preds, signal_preds
        )
