import torch

from train.data.supervised.supervised import SupervisedAframeDataset


class MultiModalSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg, psds

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            [X], waveforms = batch
            batch = self.augment(X, waveforms)
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
            X_bg, X_fg, psds = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg, psds)
        return batch

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)

        asds = psds**0.5
        asds *= 1e23
        asds = asds.float()

        X_fft = torch.fft.rfft(X)
        num_freqs = X_fft.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = torch.nn.functional.interpolate(
                asds, size=(num_freqs,), mode="linear"
            )
        inv_asds = 1 / asds
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)

        return (X, X_fft), y
