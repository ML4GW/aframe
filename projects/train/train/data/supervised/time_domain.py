import torch

from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, cross, plus):
        X_bg, X_inj, psds = super().build_val_batches(background, cross, plus)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def inject(self, X):
        X, y, psds = super().inject(X)
        X = self.whitener(X, psds)
        return X, y
