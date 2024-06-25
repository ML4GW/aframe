from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def augment(self, X):
        X, y, psds = super().augment(X)
        X = self.whitener(X, psds)
        return X, y
