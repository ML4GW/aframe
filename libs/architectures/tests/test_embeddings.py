import torch
from ml4gw.nn.resnet.resnet_1d import ResNet1D

from architectures.supervised import SupervisedTimeDomainResNet


def test_extract_embeddings():
    layers = [1, 1, 1, 1]
    original = ResNet1D(2, layers=layers, classes=1).eval()
    model = SupervisedTimeDomainResNet(
        num_ifos=2,
        sample_rate=2048,
        kernel_length=1.5,
        layers=layers,
    ).eval()

    model.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 2, 3072)

    with torch.no_grad():
        embeddings = model.extract_embeddings(x)
        expected = original(x)

        assert embeddings.shape == (2, model.fc.in_features)
        torch.testing.assert_close(model.fc(embeddings), expected)
        torch.testing.assert_close(model(x), expected)