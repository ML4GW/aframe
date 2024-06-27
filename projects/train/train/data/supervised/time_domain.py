from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_inj = self.whitener(X_inj, psds)
        return X_bg, X_inj

    def augment(self, X):
        X, y, psds = super().augment(X)
        X = self.whitener(X, psds)
        return X, y
