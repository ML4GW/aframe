import numpy as np
import torch

from bbhnet.data.transforms.transform import Transform
from ml4gw.utils.slicing import sample_kernels


# TODO: generalize to arbitrary ifos
class GlitchSampler(Transform):
    def __init__(
        self, prob: float, max_offset: int, **glitches: np.ndarray
    ) -> None:
        super().__init__()
        self.glitches = torch.nn.ParameterList()
        for ifo in glitches.values():
            param = self.add_parameter(ifo)
            self.glitches.append(param)

        self.prob = prob
        self.max_offset = max_offset

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if X.shape[1] < len(self.glitches):
            raise ValueError(
                "Can't insert glitches into tensor with {} channels "
                "using glitches from {} ifos".format(
                    X.shape[1], len(self.glitches)
                )
            )

        # sample batch indices which will be replaced with
        # a glitch independently from each interferometer
        masks = torch.rand(size=(len(self.glitches), len(X))) < self.prob
        for i, ifo in enumerate(self.glitches):
            mask = masks[i]

            # now sample from our bank of glitches for this
            # interferometer the number we want to insert
            N = mask.sum().item()
            idx = torch.randint(len(ifo), size=(N,))

            # finally sample kernels from the selected glitches.
            # Add a dummy dimension so that sample_kernels
            # doesn't think this is a single multi-channel
            # timeseries, but rather a batch of single
            # channel timeseries
            glitches = ifo[idx, None]
            glitches = sample_kernels(
                glitches,
                kernel_size=X.shape[-1],
                max_center_offset=self.max_offset,
            )

            # replace the appropriate channel in our
            # strain data with the sampled glitches
            X[mask, i] = glitches[:, 0]
        return X, y
