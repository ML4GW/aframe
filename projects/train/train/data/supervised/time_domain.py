import math
from typing import Literal

import torch
import torch.nn.functional as F
from ml4gw.transforms import Heterodyne

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
            If provided, only the top `k` chirp mass channels are chosen
            for the heterodyned timeseries. Otherwise, all chirp mass
            channels are returned.
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

    def _select_top_k(self, X):
        B, C, M, T = X.shape
        # If no top-k requested, flatten the chirp-mass dimension
        if self.top_k is None:
            return X.reshape(B, C * M, T)
        _X = []
        for b in range(B):
            x = X[b]
            avgpool = F.avg_pool1d(
                torch.abs(x.reshape(C * M, T)),
                kernel_size=31,
                stride=1,
                padding=15,
            ).reshape(C, M, T)
            if self.keep_last_n_seconds is not None:
                avgpool_snr = torch.sqrt(
                    (avgpool[..., -self.keep_last_n_samples :] ** 2).sum(dim=0)
                )
            else:
                avgpool_snr = torch.sqrt((avgpool**2).sum(dim=0))
            vals = avgpool_snr.max(dim=-1).values
            idx = torch.topk(vals, k=self.top_k).indices
            idx = idx[None, :, None].expand(C, -1, T)
            _X.append(torch.gather(x, dim=1, index=idx))
        return torch.stack(_X).reshape(B, C * self.top_k, T)

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_bg = self.heterodyne_transform(X_bg)
        X_bg = self._select_top_k(X_bg)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            inj = self.heterodyne_transform(inj)
            inj = self._select_top_k(inj)
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
        X = self._select_top_k(X)

        if self.keep_last_n_seconds is not None:
            return X[..., -self.keep_last_n_samples :], y
        else:
            return X, y
