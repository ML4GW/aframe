from unittest.mock import Mock

import numpy as np
import pytest
import torch
from train.validation import Validator


def add_method(mock, method):
    def dummy(*args, **kwargs):
        return method(mock, *args, **kwargs)

    setattr(mock, method.__name__, dummy)


class TestValidator:
    def test_steps_for_shift(self):
        mock = Mock()
        mock.duration = 20
        mock.kernel_length = 2
        mock.stride = 1.5

        steps = Validator.steps_for_shift(mock, 0)
        assert steps == 13

        steps = Validator.steps_for_shift(mock, 1)
        assert steps == 12

    def test_shift_background(self):
        x = np.arange(128)
        mock = Mock()
        mock.background = np.stack([x, -x])
        mock.sample_rate = 8

        background = Validator.shift_background(mock, 0)
        np.testing.assert_array_equal(background, mock.background)

        background = Validator.shift_background(mock, 1)
        assert background.shape == (2, 120)
        expected = np.arange(120)
        np.testing.assert_array_equal(background[0], expected)

        expected = -np.arange(8, 128)
        np.testing.assert_array_equal(background[1], expected)

        background = Validator.shift_background(mock, -1)
        assert background.shape == (2, 120)
        expected = np.arange(8, 128)
        np.testing.assert_array_equal(background[0], expected)

        expected = -np.arange(120)
        np.testing.assert_array_equal(background[1], expected)

    def test_iter_shift(self):
        x = np.arange(128)
        mock = Mock()
        mock.background = np.stack([x, -x])
        mock.sample_rate = 8
        mock.duration = 16
        mock.batch_size = 4
        mock.device = "cpu"

        mock.kernel_length = 2
        mock.kernel_size = 16

        mock.stride = 1.5
        mock.stride_size = 12

        add_method(mock, Validator.steps_for_shift)
        add_method(mock, Validator.shift_background)

        it = Validator.iter_shift(mock, 0)
        X = next(it).numpy()
        assert X.shape == (4, 2, 16)
        expected = np.arange(16)
        for i, x in enumerate(X):
            y = i * 12 + expected
            np.testing.assert_array_equal(x[0], y)
            np.testing.assert_array_equal(x[1], -y)

        X = next(it).numpy()
        assert X.shape == (4, 2, 16)
        for i, x in enumerate(X):
            y = (i + 4) * 12 + expected
            np.testing.assert_array_equal(x[0], y)
            np.testing.assert_array_equal(x[1], -y)

        X = next(it).numpy()
        assert X.shape == (2, 2, 16)
        for i, x in enumerate(X):
            y = (i + 8) * 12 + expected
            np.testing.assert_array_equal(x[0], y)
            np.testing.assert_array_equal(x[1], -y)

        with pytest.raises(StopIteration):
            next(it)

    def test_postprocess(self):
        mock = Mock()
        mock.integration_size = 2
        mock.window = torch.ones((1, 1, 2)) / 2
        mock.pool_size = 8

        preds = torch.arange(32) + 1
        preds = preds.type(torch.float32)
        post = Validator.postprocess(mock, preds).numpy()
        assert post.shape == (4,)

        expected = 8 * np.arange(1, 5) - 0.5
        np.testing.assert_array_equal(post, expected)

    def test_inject(self):
        mock = Mock()
        mock._injection_step = 4
        mock._injection_idx = 0
        mock.kernel_size = 16

        x = torch.arange(16)
        x = torch.stack([x, x])
        X = torch.stack([x + i for i in range(13)])
        X[:, 1] *= -1

        waveforms = torch.zeros((3, 2, 32))
        waveforms[:, :, 8:-8] += torch.arange(3)[:, None, None] + 1
        mock.waveforms = waveforms

        injected = Validator.inject(mock, X).numpy()
        assert injected.shape == (3, 2, 16)
        expected = np.arange(16)
        for i, x in enumerate(injected):
            y = expected + 5 * i + 1
            np.testing.assert_array_equal(x[0], y)

            y = -expected - 3 * i + 1
            np.testing.assert_array_equal(x[1], y)
