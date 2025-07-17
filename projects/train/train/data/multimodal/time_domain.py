import torch
from train.data.supervised.supervised import SupervisedAframeDataset

class TimeDomainMultimodalAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)

        #whiten backgtound
        X_bg = self.whitener(X_bg, psds)
        X_bg = self.qtransform(X_bg)

        #whiten injection
        X_fg = []
        for inj in X_inj
            inj = self.whitener(inj, psds)
            X_fg.append(self.qtransform(inj))

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)

        #whiten input
        X = self.whitener(X, psds)

        return X, y, psds
