import torch
import torch.nn.functional as F
from train.data.supervised.supervised import SupervisedAframeDataset

class FrequencyDomainMultimodalAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        
        #apply whitening with low high filters
        X_bg_low = self.whitener(X_bg, psds, lowpass=self.hparams.lowpass, highpass=None)
        X_bg_high = self.whitener(X_bg, psds, lowpass=None, highpass=self.hparams.highpass)

        X_fg_low, X_fg_high = [], []
        for inj in X_inj:
            inj_low = self.whitener(inj, psds, lowpass=self.hparams.lowpass, highpass=None)
            inj_high = self.whitener(inj, psds, lowpass=None, highpass=self.hparams.highpass)
            X_fg_low.append(inj_low)
            X_fg_high.append(inj_high)

        X_fg_low = torch.stack(X_fg_low)
        X_fg_high = torch.stack(X_fg_high)

        # FFT for full-band
        asds = psds.sqrt() * 1e23
        asds = asds.float()

        X_bg_fft = torch.fft.rfft(X_bg)
        X_fg_fft = torch.fft.rfft(X_inj)

        if asds.shape[-1] != X_bg_fft.shape[-1]:
            asds = F.interpolate(asds, size=X_bg_fft.shape[-1], mode="linear", align_corners=False)
        
        inv_asds = 1 / asds
        
        if X_fg_fft.real.ndim == 4:
            inv_asds_inj = inv_asds.unsqueeze(0).expand(X_fg_fft.shape[0], -1, -1, -1)
        else:
            inv_asds_inj = inv_asds

        X_bg_fft = torch.cat([X_bg_fft.real, X_bg_fft.imag, inv_asds], dim=1)
        X_fg_fft = torch.cat([X_fg_fft.real, X_fg_fft.imag, inv_asds_inj], dim=2)

        return (X_bg_low, X_bg_high, X_bg_fft), (X_fg_low, X_fg_high, X_fg_fft), psds

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)

        #apply low/high split
        X_low = self.whitener(X, psds, lowpass=self.hparams.lowpass, highpass=None)
        X_high = self.whitener(X, psds, lowpass=None, highpass=self.hparams.highpass)

        #fft feature for full-band
        asds = psds.sqrt() * 1e23
        asds = asds.float()

        X_fft = torch.fft.rfft(X)
        if asds.shape[-1] != X_fft.shape[-1]:
            asds = F.interpolate(asds, size=X_fft.shape[-1], mode="linear", align_corners=False)

        inv_asds = 1 / asds
        X_fft = torch.cat([X_fft.real, X_fft.imag, inv_asds], dim=1)

        return (X_low, X_high, X_fft), y
