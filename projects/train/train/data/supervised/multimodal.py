import torch

from train.data.supervised.supervised import SupervisedAframeDataset


class MultiModalSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_bg_fft = self.compute_frequency_domain_data(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        X_fg_fft = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)
            X_fg_fft.append(self.compute_frequency_domain_data(inj, psds))

        X_fg = torch.stack(X_fg)
        X_fg_fft = torch.stack(X_fg_fft)

        asds = psds**0.5
        asds *= 1e23
        asds = asds.float()

        return (X_bg, X_bg_fft), (X_fg, X_fg_fft)

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
            (X, X_fft), y = self.augment(X, waveforms)
            batch = (X, X_fft, y)
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
            (X_bg, X_bg_fft), (X_fg, X_fg_fft) = self.build_val_batches(
                background, signals
            )
            batch = (shift, X_bg, X_fg, X_bg_fft, X_fg_fft)
        return batch

    def compute_frequency_domain_data(self, X, psds):
        asds = psds**0.5

        freqs = torch.fft.rfftfreq(X.shape[-1], d=1 / self.hparams.sample_rate)
        num_freqs = len(freqs)

        asds = torch.nn.functional.interpolate(
            asds,
            size=(num_freqs,),
            mode="linear",
        )
        mask = torch.ones_like(freqs, dtype=torch.bool)
        if self.hparams.highpass is not None:
            mask &= freqs > self.hparams.highpass
        if self.hparams.lowpass is not None:
            mask &= freqs < self.hparams.lowpass
        asds = asds[:, :, mask]

        asds *= 1e23
        asds = asds.float()
        inv_asd = 1 / asds

        X_fft = torch.fft.rfft(X, dim=-1)
        X_fft = X_fft[:, :, mask]
        X_fft = torch.cat([X_fft.real, X_fft.imag, inv_asd], dim=1)

        return X_fft

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)
        X_fft = self.compute_frequency_domain_data(X, psds)

        return (X, X_fft), y
