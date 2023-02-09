from typing import Optional

import torch

from ml4gw.transforms import Whitening


class Preprocessor(torch.nn.Module):
    """
    Module for encoding BBHNet preprocessing procedure.
    Very simple wrapper for now, but encoding it this
    way to accommodate potentially more complex preprocessing
    in the future.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        fduration: Optional[float] = None,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.whitener = Whitening(
            num_ifos,
            sample_rate,
            fduration=fduration,
        )

    def forward(self, x):
        x = self.whitener(x)
        return x
