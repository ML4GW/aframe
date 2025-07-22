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
        super().__init__(*args, swap_prob=swap_prob, mute_prob=mute_prob, **kwargs)

    @torch.no_grad()
    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)

        X_low = self.whitener(X, psds, lowpass=self.hparams.lowpass, highpass=None)
        X_high = self.whitener(X, psds, lowpass=None, highpass=self.hparams.highpass)

        X_fft = torch.fft.rfft(X)
        asds = psds.sqrt() * 1e23
        asds = asds.float()
        if asds.shape[-1] != X_fft.shape[-1]:
            asds = F.interpolate(asds, size=X_fft.shape[-1], mode="linear", align_corners=False)
        inv_asds = 1 / asds
        X_fft = torch.cat([X_fft.real, X_fft.imag, inv_asds], dim=1)

        return {
            "psd_low": X_low,
            "psd_high": X_high,
            "fft": X_fft,
            "label": y,
        }

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            X, waveforms = batch
            X = self.augment(X, waveforms)
            return X, X["label"]

        elif self.trainer.validating or self.trainer.sanity_checking:
            [background, _, timeslide_idx], [signals] = batch
            shift = self.timeslides[timeslide_idx].shift_size

            X_bg, X_inj, psds = super().build_val_batches(background, signals)

            X_bg_low = self.whitener(X_bg, psds, highpass=self.hparams.highpass, lowpass=self.hparams.lowpass)
            X_bg_high = self.whitener(X_bg, psds, highpass=self.hparams.lowpass, lowpass=None)

            X_fg_low, X_fg_high = [], []
            for inj in X_inj:
                inj_low = self.whitener(inj, psds, lowpass=self.hparams.lowpass, highpass=self.hparams.highpass)
                inj_high = self.whitener(inj, psds, lowpass=None, highpass=self.hparams.lowpass)
                X_fg_low.append(inj_low)
                X_fg_high.append(inj_high)
            X_fg_low = torch.stack(X_fg_low)
            X_fg_high = torch.stack(X_fg_high)

            X_bg_fft = torch.fft.rfft(X_bg)
            X_fg_fft = torch.fft.rfft(X_inj)
            asds = psds.sqrt() * 1e23
            if asds.shape[-1] != X_bg_fft.shape[-1]:
                asds = F.interpolate(asds, size=X_bg_fft.shape[-1], mode="linear", align_corners=False)
            inv_asds = 1 / asds
            if X_fg_fft.real.ndim == 4:
                inv_asds_inj = inv_asds.unsqueeze(0).expand(X_fg_fft.shape[0], -1, -1, -1)
            else:
                inv_asds_inj = inv_asds
            X_bg_fft = torch.cat([X_bg_fft.real, X_bg_fft.imag, inv_asds], dim=1)
            X_fg_fft = torch.cat([X_fg_fft.real, X_fg_fft.imag, inv_asds_inj], dim=2)

            return (
                shift,
                {
                    "psd_low": X_bg_low,
                    "psd_high": X_bg_high,
                    "fft": X_bg_fft,
                },
                {
                    "psd_low": X_fg_low,
                    "psd_high": X_fg_high,
                    "fft": X_fg_fft,
                },
                psds,
            )

        return batch

