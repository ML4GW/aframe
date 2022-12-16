import pytest
import torch

from bbhnet.data.transforms.standard_scaler import StandardScalerTransform


def test_standard_scaler_transform(num_ifos):
    scaler = StandardScalerTransform(num_ifos)
    assert len(list(scaler.parameters())) == 2

    for i, param in enumerate([scaler.mean, scaler.std]):
        assert param.ndim == 1
        assert len(param) == num_ifos
        assert (param == i).all()

    x = torch.arange(10).type(torch.float32)
    X = torch.stack([x + i for i in range(num_ifos)])
    scaler.fit(X)

    expected_mean = torch.Tensor([4.5 + i for i in range(num_ifos)])
    assert (scaler.mean == expected_mean).all()
    assert (scaler.std == (110 / 12) ** 0.5).all()

    batch = torch.stack([X, X])
    y = scaler(batch)
    assert (y.mean(axis=-1) == 0).all()
    assert (y.std(axis=-1) == 1).all()

    with pytest.raises(ValueError) as exc_info:
        scaler.fit(batch)
    assert str(exc_info.value).startswith("Expected background")
    assert str(exc_info.value).endswith("but found 3")

    # can't sub-slice the number of ifos. We could
    # add more, but I'll let the other tests check
    # for this and save ourselves some boilerplate
    if num_ifos == 1:
        return

    for bad_batch in [X, batch[:, :1]]:
        with pytest.raises(ValueError):
            scaler(bad_batch)
