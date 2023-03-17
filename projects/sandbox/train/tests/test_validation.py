import numpy as np
import pytest
import torch
from train import validation


def test_background_recall():
    metric = validation.BackgroundRecall(10, 10, 3)
    background = torch.zeros((31,))

    # after pooling, the top 3 values should come
    # out to 2, 1.5, and 0.8
    background[3] = 1
    background[4] = 1.5
    background[10] = 2
    background[14] = 0.5
    background[18] = 0.9
    background[22] = 0.8
    background[30] = 3  # this will get trimmed

    signal = torch.arange(0.5, 2.5, 0.1)
    scores = metric(background, None, signal).cpu().numpy()
    assert len(scores) == 3
    assert all([i == j for i, j in zip(scores, metric.values)])

    np.testing.assert_allclose(scores, [0.25, 0.5, 0.85])


def test_auroc():
    auroc = validation.MultiThresholdAUROC([0.01, 0.1, 1])
    signal_preds = 10 + 0.1 * torch.randn(1000)
    background_preds = 0.1 * torch.randn(1000)

    scores = auroc.call(signal_preds, background_preds)
    expected = torch.tensor([10 ** (i - 2) for i in range(3)])

    # account for the extra sample that
    # gets in due to the <=
    expected[:-1] += 1 / 1000
    assert torch.allclose(scores, expected, rtol=1e-6)

    scores = auroc.call(background_preds, signal_preds)
    assert (scores == 0).all().item()

    # now build an arbitrary ROC curve and
    # make sure the areas are correct. Adding
    # in a coupple small factors of 10000 to
    # account for imprecision at the integration
    # boundaries that I'm too lazy to solve for
    # exactly, sue me
    signal_preds[:100] += 30
    background_preds[:50] += 35
    area0 = 0.01 * 0.1 + 1 / 10000

    signal_preds[100:500] += 20
    background_preds[50:200] += 25
    area1 = area0 + 0.1 * 0.04 + 0.5 * 0.05 + 4 / 10000

    signal_preds[500:700] += 10
    background_preds[200:800] += 15
    area2 = area1 + 0.5 * 0.1 + 0.7 * 0.6 + 0.2 - 5 / 10000

    scores = auroc.call(signal_preds, background_preds)
    expected = torch.tensor([area0, area1, area2])
    assert torch.allclose(scores, expected, rtol=1e-6)

    # ensure that this isn't an artifact of
    # the ordering by shuffling and making
    # sure things still match
    idx = torch.randperm(1000)
    signal_preds = signal_preds[idx]
    idx = torch.randperm(1000)
    background_preds = background_preds[idx]
    scores = auroc.call(signal_preds, background_preds)
    assert torch.allclose(scores, expected, rtol=1e-6)

    # now verify that constant outputs
    # will produce a score signifying
    # random predictions
    constants = torch.zeros((1000,))
    scores = auroc.call(constants, constants)

    # ensure that "random" scores are roughly
    # equal to the area under the y=x line up
    # to the max fpr value
    expected = [0.5 * 10 ** (2 * (i - 2)) for i in range(3)]
    expected = torch.tensor(expected)
    assert torch.allclose(scores, expected, rtol=0.1)


def test_glitch_recall():
    metric = validation.GlitchRecall([0.5, 0.75, 1])
    glitches = torch.arange(11).type(torch.float32)
    signal = torch.arange(4, 12).type(torch.float32)

    scores = metric(None, glitches, signal).cpu().numpy()
    assert len(scores) == 3
    assert all([i == j for i, j in zip(scores, metric.values)])

    np.testing.assert_allclose(scores, [7 / 8, 4 / 8, 2 / 8])


def test_make_background():
    x = torch.arange(2 * 32).reshape(2, 32).type(torch.float32)
    batched = validation.make_background(x, 10, 5)
    assert batched.shape == (5, 2, 10)

    for i in range(5):
        for j in range(2):
            start = i * 5 + j * 32
            expected = torch.arange(start, start + 10)
            assert (batched[i, j] == expected).all().item()


def test_make_glitches():
    background = 2 + torch.arange(2 * 32).reshape(2, 32).type(torch.float32)
    background = validation.make_background(background, 10, 5)

    glitches = [torch.zeros((9, 12)), torch.ones((11, 12))]

    with pytest.raises(ValueError) as exc:
        validation.make_glitches(glitches, background, 1)
    assert str(exc.value).startswith("There are more coincident")

    # put wrong values at the edges to make sure
    # that these get appropriate sliced out
    glitches[0][:, [0, -1]] = 1
    glitches[1][:, [0, -1]] = 0

    dataset = validation.make_glitches(glitches, background, 0.5)
    assert dataset.shape == (16, 2, 10)

    # first 9 - 4 = 5 should be just H1
    assert (dataset[:5, 0] == 0).all().item()
    assert (dataset[:5, 1] != 0).all().item()

    # next 11 - 4 = 7 should be just L1
    assert (dataset[5:12, 0] != 1).all().item()
    assert (dataset[5:12, 1] == 1).all().item()

    # remaining (0.5**2) * 16 = 4 should be coincident
    assert (dataset[12:, 0] == 0).all().item()
    assert (dataset[12:, 1] == 1).all().item()
