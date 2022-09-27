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

        masks = torch.rand(size=(len(self.glitches), len(X))) < self.prob
        for i, ifo in enumerate(self.glitches):
            mask = masks[i]
            N = mask.sum().item()
            idx = torch.randint(len(ifo), size=(N,))

            glitches = ifo[idx]
            glitches = sample_kernels(
                glitches[:, None],
                kernel_size=X.shape[-1],
                max_center_offset=self.max_offset,
            )
            X[mask] = glitches
        return X, y
