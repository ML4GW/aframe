import torch

from ml4gw.utils.slicing import sample_kernels
from train.data.base import BaseAframeDataset

Tensor = torch.Tensor


class AutoencoderAframeDataset(BaseAframeDataset):
    @torch.no_grad()
    def augment(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
        Looks a lot like the supervised case, but we inject
        on _all_ samples and don't do any swapping or muting.
        """

        X, psds = self.psd_estimator(X)

        X = self.inverter(X)
        X = self.reverser(X)
        *params, polarizations, mask = self.waveform_sampler(X)

        N = len(params[0])
        snrs = self.snr_sampler(N).to(X.device)
        responses = self.projector(*params, snrs, psds[mask], **polarizations)

        responses = self.pad_waveforms(responses, X.size(-1))
        kernels = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True
        )
        X += kernels
        X = self.whitener(X, psds)
        return X, psds
