from typing import Optional

import torch

from ml4gw.utils.slicing import sample_kernels
from train import augmentations as aug
from train.data.base import BaseAframeDataset


class SupervisedAframeDataset(BaseAframeDataset):
    def __init__(
        self,
        *args,
        swap_frac: Optional[float] = None,
        mute_frac: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        upweight, cross_term = 1, 1
        if swap_frac is not None and 0 < swap_frac < 1:
            self.swapper = aug.ChannelSwapper(swap_frac)
            upweight -= swap_frac
            cross_term *= swap_frac
        elif swap_frac is not None:
            raise ValueError(
                f"swap_frac must be between 0 and 1, got {swap_frac}"
            )
        else:
            cross_term *= 0
            self.swapper = None

        if mute_frac is not None and 0 < mute_frac < 1:
            self.muter = aug.ChannelMuter(mute_frac)
            upweight -= mute_frac
            cross_term *= mute_frac
        elif mute_frac is not None:
            raise ValueError(
                f"mute_frac must be between 0 and 1, got {mute_frac}"
            )
        else:
            cross_term *= 0
            self.muter = None

        upweight += cross_term
        self.hparams.waveform_prob /= upweight
        if self.hparams.waveform_prob > 1:
            raise ValueError(
                "Waveform prob is {} after upweighting, must "
                "remain <1".format(self.hparams.waveform_prob)
            )

    @torch.no_grad()
    def augment(self, X):
        X, psds = self.psd_estimator(X)

        X = self.inverter(X)
        X = self.reverser(X)
        *params, polarizations, mask = self.waveform_sampler(X)

        N = len(params[0])
        snrs = self.snr_sampler(N).to(X.device)
        responses = self.projector(*params, snrs, psds[mask], **polarizations)
        kernels = sample_kernels(
            responses,
            kernel_size=X.size(-1),
            max_center_offset=self.pad_size,
            coincident=True,
        )

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses and whiten
        X[mask] += kernels
        X = self.whitener(X, psds)

        # make labels, turning off injection mask where
        # we swapped or muted
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1
        return X, y
