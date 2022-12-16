import torch

from bbhnet.data.transforms.transform import Transform


class StandardScalerTransform(Transform):
    def __init__(self, num_ifos: int):
        super().__init__()
        self.mean = self.add_parameter(torch.zeros([num_ifos]))
        self.std = self.add_parameter(torch.ones([num_ifos]))

    def fit(self, X: torch.Tensor) -> None:
        if X.ndim != 2:
            raise ValueError(
                "Expected background used to fit WhiteningTransform "
                "to have 2 dimensions, but found {}".format(X.ndim)
            )

        self.set_value(self.mean, X.mean(axis=-1))
        self.set_value(self.std, X.std(axis=-1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3 or X.shape[1] != len(self.mean):
            raise ValueError(
                "Can't transform tensor of shape {}"
                "using StandardScaler for {} channels".format(
                    X.shape, len(self.mean)
                )
            )

        X = X.transpose(1, 2)
        X = (X - self.mean) / self.std
        return X.transpose(2, 1)
