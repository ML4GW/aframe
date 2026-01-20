from typing import Optional
import torch
import torch.nn.functional as F
from train.data.supervised.supervised import SupervisedAframeDataset


class MultimodalSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        *args,
        swap_prob: Optional[float] = None,
        mute_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, swap_prob=swap_prob, mute_prob=mute_prob, **kwargs
        )

    @torch.no_grad()
    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)

        psds = psds

        X_whitened = self.whitener(X, psds)
        X_low = self.whitener(
            X,
            psds,
            highpass=self.hparams.highpass,
            lowpass=self.hparams.lowpass,
        ).float()
        X_high = self.whitener(
            X, psds, highpass=self.hparams.lowpass, lowpass=None
        ).float()

        X_fft = torch.fft.rfft(X_whitened, dim=-1)
        asds = psds**0.5

        freqs = torch.fft.rfftfreq(
            X_whitened.shape[-1], d=1 / self.hparams.sample_rate
        )
        num_freqs = len(freqs)

        asds = torch.nn.functional.interpolate(
            asds,
            size=(num_freqs,),
            mode="linear",
        )
        mask = torch.ones_like(freqs, dtype=torch.bool)
        if self.hparams.highpass is not None:
            mask &= freqs > self.hparams.highpass
        # if self.hparams .lowpass is not None:
        #    mask &= freqs < self.hparams.lowpass

        asds = asds[:, :, mask]
        asds *= 1e23
        inv_asds = 1 / asds

        X_fft = X_fft[:, :, mask]
        X_fft = torch.cat([X_fft.real, X_fft.imag, inv_asds], dim=1).float()
        if torch.isnan(X_low).any() or torch.isinf(X_low).any():
            raise ValueError("NaN or Inf in X_low")
        if torch.isnan(X_fft).any() or torch.isinf(X_fft).any():
            raise ValueError("NaN or Inf in X_fft")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("NaN or Inf in y")

        return X_low, X_high, X_fft, y.float()

    def on_after_batch_transfer(self, batch, _):
        """
        Perform on-device preprocessing after transferring batch to device.

        Augments data during training and injects signals into background during validation,
        performing whitening and FFT-based preprocessing.
        """
        if self.trainer.training:
            # Training mode: perform random augmentations using waveforms
            [X], waveforms = batch
            return self.augment(X, waveforms)

        elif self.trainer.validating or self.trainer.sanity_checking:
            # Validation mode: prepare signal-injected validation batches
            [background, _, timeslide_idx], [signals] = batch
            if isinstance(timeslide_idx, torch.Tensor):
                timeslide_idx = timeslide_idx[0].item()
            shift = float(self.timeslides[timeslide_idx].shift_size)

            # Build validation inputs and corresponding PSDs
            X_bg, X_inj, psds = super().build_val_batches(background, signals)

            # Background: low/high-passed and FFT-processed
            X_bg_whitened = self.whitener(X_bg, psds)
            X_bg_low = self.whitener(
                X_bg,
                psds,
                highpass=self.hparams.highpass,
                lowpass=self.hparams.lowpass,
            ).float()
            X_bg_high = self.whitener(
                X_bg, psds, highpass=self.hparams.lowpass, lowpass=None
            ).float()
            X_bg_fft = torch.fft.rfft(X_bg_whitened)

            # Foreground: process injected signals similarly
            X_fg_whitened, X_fg_low, X_fg_high = [], [], []
            for inj in X_inj:
                X_fg_low.append(
                    self.whitener(
                        inj,
                        psds,
                        lowpass=self.hparams.lowpass,
                        highpass=self.hparams.highpass,
                    ).float()
                )
                X_fg_high.append(
                    self.whitener(
                        inj, psds, lowpass=None, highpass=self.hparams.lowpass
                    ).float()
                )
                X_fg_whitened.append(self.whitener(inj, psds))
            X_fg_low = torch.stack(X_fg_low)
            X_fg_high = torch.stack(X_fg_high)
            X_fg_whitened = torch.stack(X_fg_whitened, dim=0)
            X_fg_fft = torch.fft.rfft(X_fg_whitened)

            asds = psds**0.5
            asds *= 1e23
            asds = asds.float()
            num_freqs = X_fg_fft.shape[-1]
            if asds.shape[-1] != num_freqs:
                asds = F.interpolate(asds, size=(num_freqs,), mode="linear")
            inv_asds = 1 / asds

            X_bg_fft = torch.cat(
                [X_bg_fft.real, X_bg_fft.imag, inv_asds], dim=1
            ).float()
            inv_asds = inv_asds.unsqueeze(0).repeat(5, 1, 1, 1)
            X_fg_fft = torch.cat(
                [X_fg_fft.real, X_fg_fft.imag, inv_asds], dim=2
            ).float()

            # Return data grouped into background and injected signal components
            return (
                shift,
                X_bg_low,
                X_bg_high,
                X_bg_fft,
                X_fg_low,
                X_fg_high,
                X_fg_fft,
                psds,
            )

        # Default: return batch unchanged
        return batch