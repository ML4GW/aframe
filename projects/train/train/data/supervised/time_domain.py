import math
from typing import Literal

import torch
from ml4gw.transforms import Heterodyne
from utils.augmentation import select_top_k

from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        return X, y


class HeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    """
    A derived class from BaseAframeDataset and SupervisedAframeDataset, it
    applies heterodyning to strain data and returns heterodyned timeseries
    for loading data to train Aframe models. If `keep_last_n_seconds` is
    passed, returns only the final portion of the heterodyned strain.

    Args:
        chirp_mass_low (float):
            Lower bound of chirp mass range (in solar masses).
        chirp_mass_high (float):
            Upper bound of chirp mass range (in solar masses).
        num_chirp_masses (int):
            Number of chirp mass samples to generate.
        chirp_mass_spacing (Literal["linear", "log"]):
            Spacing of chirp mass grid. Use "linear" for evenly spaced
            values or "log" for logarithmic spacing.
        keep_last_n_seconds (float):
            If provided, only the last `n` seconds of the kernel_length are
            returned. Otherwise, the full kernel_length is returned.
        top_k (int):
            If provided, the top `k` chirp mass channels are selected based on
            their absolute amplitude, corresponding to the `k` matching chirp
            masses with. If `None`, all chirp mass channels are returned.
    """

    def __init__(
        self,
        chirp_mass_low: float = 1.0,
        chirp_mass_high: float = 2.5,
        num_chirp_masses: int = 100,
        chirp_mass_spacing: Literal["linear", "log"] = "log",
        keep_last_n_seconds: float = None,
        top_k: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_grid = self._create_chirp_mass_grid(
            chirp_mass_low,
            chirp_mass_high,
            num_chirp_masses,
            chirp_mass_spacing,
        )

        self.keep_last_n_seconds = keep_last_n_seconds
        self.top_k = top_k
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        else:
            self.keep_last_n_samples = None

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.heterodyne_transform = Heterodyne(
            sample_rate=self.hparams.sample_rate,
            kernel_length=self.hparams.kernel_length,
            chirp_mass=self.chirp_mass_grid,
            return_type="time",
        )

    def _create_chirp_mass_grid(
        self,
        chirp_mass_low: float,
        chirp_mass_high: float,
        num_chirp_masses: int,
        chirp_mass_spacing: Literal["linear", "log"],
    ) -> torch.Tensor:
        if chirp_mass_spacing == "linear":
            return torch.linspace(
                chirp_mass_low, chirp_mass_high, num_chirp_masses
            )
        elif chirp_mass_spacing == "log":
            return torch.logspace(
                math.log10(chirp_mass_low),
                math.log10(chirp_mass_high),
                num_chirp_masses,
            )
        else:
            raise ValueError(
                f"Invalid chirp mass spacing: {chirp_mass_spacing}"
            )

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_bg = self.heterodyne_transform(X_bg)
        if self.top_k is not None:
            X_bg = select_top_k(
                X_bg, self.top_k, keep_last_n_samples=self.keep_last_n_samples
            )
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            inj = self.heterodyne_transform(inj)
            if self.top_k is not None:
                inj = select_top_k(
                    inj,
                    self.top_k,
                    keep_last_n_samples=self.keep_last_n_samples,
                )
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)

        if self.keep_last_n_seconds is not None:
            return X_bg[..., -self.keep_last_n_samples :], X_fg[
                ..., -self.keep_last_n_samples :
            ]
        else:
            return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        X = self.heterodyne_transform(X)
        if self.top_k is not None:
            X = select_top_k(
                X, self.top_k, keep_last_n_samples=self.keep_last_n_samples
            )

        if self.keep_last_n_seconds is not None:
            return X[..., -self.keep_last_n_samples :], y
        else:
            return X, y


class TimeDomainSupervisedRegressionDataset(SupervisedAframeDataset):
    def on_before_batch_transfer(self, batch, _):
        if self.trainer.training and self.waveforms_from_disk:
            X, waveforms = batch
            waveforms, new_signal_idx = self.slice_waveforms(waveforms)
            batch = X, waveforms, new_signal_idx
        return batch

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            if self.waveforms_from_disk:
                [batch], waveforms, new_signal_idx = batch
                batch = self.inject(batch, waveforms, new_signal_idx)
            else:
                [batch] = batch
                batch = self.inject(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg, mu = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg, mu)
        return batch

    @torch.no_grad()
    def inject(self, X, waveforms=None, new_signal_idx=None):
        if self.waveforms_from_disk and waveforms is None:
            raise ValueError(
                "Waveforms should be passed to the `inject` method "
                "if waveforms are being loaded from disk, got None"
            )

        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)
        # sample enough waveforms to do true injections,
        # swapping, and muting

        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob

        dec, psi, phi = self.sample_extrinsic(X[mask])
        # If we're loading waveforms from disk, we can
        # slice out the ones we want.
        # If not, we're generating them on the fly.
        if self.waveforms_from_disk:
            # TODO: Can we just use `mask` to slice out the
            # waveforms we want here? Copying this from the
            # old `WaveformSampler` in case it handles edge
            # cases I'm not thinking of
            N = mask.sum().item()
            idx = torch.randperm(waveforms.shape[0])[:N]
            waveforms = waveforms[idx].to(X.device).float()
            hc, hp = waveforms[:, 0], waveforms[:, 1]
        else:
            hc, hp = self.waveform_sampler.sample(X[mask])

        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
        responses = self.projector(
            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
        )
        # If we're loading waveforms from disk, we'll have sliced
        # the waveforms already in `on_before_batch_transfer`
        if not self.waveforms_from_disk:
            responses, new_signal_idx = self.slice_waveforms(responses)
        kernels, idx = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True, return_idx=True
        )
        mu = new_signal_idx / kernels.shape[-1]

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        swap_indices = mute_indices = []
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses
        X[mask] += kernels

        # make labels, turning off injection mask where
        # we swapped or muted
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1
        mu = mu[~swap_indices * ~mute_indices]

        X = self.whitener(X, psds)
        return X, y, mu

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)

        # Get the position of each signal in the injected dataset,
        # normalized by the length of the full kernel
        kernel_size = X_bg.shape[-1]
        if self.hparams.num_valid_views == 1:
            step = 0
        else:
            step = kernel_size - self.left_pad_size - self.right_pad_size
            step /= self.hparams.num_valid_views - 1

        mu = [
            self.left_pad_size + i * step
            for i in range(self.hparams.num_valid_views)
        ]
        mu = torch.Tensor(mu) / kernel_size

        batch_size = X_bg.shape[0]
        mu = mu.unsqueeze(-1).repeat(1, batch_size)

        return X_bg, X_fg, mu