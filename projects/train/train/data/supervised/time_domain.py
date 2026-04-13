import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset
from ml4gw.transforms import Heterodyne

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
    for loading data to train Aframe models. If `keep_last_n_seconds` is passed,
    returns only the final portion of the heterodyned strain.

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
            If > 0, only keep the last `n` seconds of the kernel_length. If 0,
            keep the full kernel_length.
    """

    def __init__(
        self, 
        chirp_mass_low: float = 1.0, 
        chirp_mass_high: float = 2.5, 
        num_chirp_masses: int = 100, 
        chirp_mass_spacing: Literal["linear", "log"] = "log", 
        keep_last_n_seconds: float = 0.0, 
        *args, 
        **kwargs):
        super().__init__(*args, **kwargs)

        self.chirp_mass_grid = self._create_chirp_mass_grid(
            chirp_mass_low,
            chirp_mass_high,
            num_chirp_masses,
            chirp_mass_spacing,
        )

        self.keep_last_n_samples = int(
            keep_last_n_seconds * self.hparams.sample_rate
        )

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.heterodyne_transform = Heterodyne(
            sample_rate=int(self.hparams.sample_rate), 
            kernel_length=int(self.hparams.kernel_length),
            chirp_mass=self.chirp_mass_grid,
            return_type="time"
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
        _B_bg, _C_bg, _M_bg, _T_bg = X_bg.shape
        X_bg = X_bg.view(_B_bg, _C_bg*_M_bg, _T_bg)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            inj = self.heterodyne_transform(inj)
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)
        _V_fg, _B_fg, _C_fg, _M_fg, _T_fg = X_fg.shape
        X_fg = X_fg.view(_V_fg, _B_fg, _C_fg*_M_fg, _T_fg)

        if self.keep_last_n_samples > 0:
            return X_bg[..., -self.keep_last_n_samples:], X_fg[..., -self.keep_last_n_samples:]
        else:
            return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C*_M, _T)

        if self.keep_last_n_samples > 0:
            return X[..., -self.keep_last_n_samples:], y
        else:
            return X, y
