import torch
from ml4gw.utils.slicing import sample_kernels

from train.data.supervised.supervised import SupervisedAframeDataset


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


class TimeDomainSupervisedRegressionDataset(SupervisedAframeDataset):
    def on_before_batch_transfer(self, batch, _):
        if self.trainer.training and self.waveforms_from_disk:
            X, waveforms = batch
            waveforms = self.slice_waveforms(waveforms)
            batch = X, waveforms
        return batch

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            if self.waveforms_from_disk:
                [batch], waveforms = batch
                batch = self.inject(batch, waveforms)
            else:
                [batch] = batch
                batch = self.inject(batch)
            X, (y, mu) = batch
            batch = (X, y, mu)
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
            X_bg, X_fg, mu = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg, mu)
        return batch

    @torch.no_grad()
    def inject(self, X, waveforms=None):
        if self.waveforms_from_disk and waveforms is None:
            raise ValueError(
                "Waveforms should be passed to the `inject` method "
                "if waveforms are being loaded from disk, got None"
            )

        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)
        # sample enough waveforms to do true injections,
        # swapping, and muting

        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob

        dec, psi, phi = self.sample_extrinsic(X[mask])
        # If we're loading waveforms from disk, we can
        # slice out the ones we want.
        # If not, we're generating them on the fly.
        if self.waveforms_from_disk:
            # TODO: Can we just use `mask` to slice out the
            # waveforms we want here? Copying this from the
            # old `WaveformSampler` in case it handles edge
            # cases I'm not thinking of
            N = mask.sum().item()
            idx = torch.randperm(waveforms.shape[0])[:N]
            waveforms = waveforms[idx].to(X.device).float()
            hc, hp = waveforms[:, 0], waveforms[:, 1]
        else:
            hc, hp = self.waveform_sampler.sample(X[mask])

        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
        responses = self.projector(
            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
        )
        # If we're loading waveforms from disk, we'll have sliced
        # the waveforms already in `on_before_batch_transfer`
        if not self.waveforms_from_disk:
            responses = self.slice_waveforms(responses)
        kernels, idx = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True, return_idx=True
        )
        mu = self.new_signal_idx - idx - self.filter_size / 2
        mu /= self.hparams.kernel_length * self.hparams.sample_rate
        mu = mu.to(X.device)

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        swap_indices = []
        mute_indices = []
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses
        X[mask] += kernels

        # make labels, turning off injection mask where
        # we swapped or muted
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1
        mu[swap_indices] = -1
        mu[mute_indices] = -1
        mu = mu[mu >= 0]

        X = self.whitener(X, psds)
        return X, (y, mu)

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)

        # Get the position of each signal in the injected dataset,
        # normalized by the length of the whitened kernel
        kernel_size = X_bg.shape[-1]
        if self.hparams.num_valid_views == 1:
            step = 0
        else:
            # Account for filter size because X_bg is whitened
            step = (
                kernel_size
                - self.left_pad_size
                - self.right_pad_size
                + self.filter_size
            )
            step /= self.hparams.num_valid_views - 1

        mu = [
            self.left_pad_size - self.filter_size // 2 + i * step
            for i in range(self.hparams.num_valid_views)
        ]
        mu = torch.Tensor(mu).to(X_bg.device) / kernel_size

        batch_size = X_bg.shape[0]
        mu = mu.unsqueeze(-1).repeat(1, batch_size)

        return X_bg, X_fg, mu
