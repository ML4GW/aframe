from unittest.mock import Mock, call, patch

import numpy as np
import pytest
import torch
from train.validation import Validator


# dummy psd estimator and whitener
# TODO: implement local whitener in ml4gw
@pytest.fixture
def psd_estimator():
    sample_rate = 8
    background_length = 2

    def f(X):
        size = background_length * sample_rate
        splits = [size, X.shape[-1] - size]
        background, X = torch.split(X, splits, dim=-1)
        return X, background

    return f


@pytest.fixture
def whitener():
    def f(x, y):
        return x

    return f


def add_method(mock, method):
    def dummy(*args, **kwargs):
        return method(mock, *args, **kwargs)

    setattr(mock, method.__name__, dummy)


class TestValidator:
    @pytest.fixture
    def background(self):
        x = np.arange(128)
        return np.stack([x, -x])[None]

    @pytest.fixture
    def waveforms(self):
        waveforms = torch.zeros((3, 2, 32))
        waveforms[:, :, 8:-8] += torch.arange(3)[:, None, None] + 1
        return waveforms

    @pytest.fixture
    def model(self):
        class Slicer(torch.nn.Module):
            def forward(self, X):
                return X[:, 0, -1:]

        return Slicer()

    def test_init(self, background, waveforms, psd_estimator, whitener):
        tracker = Mock()

        validator = Validator(
            tracker,
            background,
            waveforms,
            psd_estimator=psd_estimator,
            whitener=whitener,
            sample_rate=8,
            stride=0.5,
            injection_stride=4,
            snr_thresh=0,
            highpass=32,
            kernel_length=4,
            batch_size=8,
            pool_length=4,
            integration_length=1,
            livetime=32,
            shift=1,
            max_fpr=1e-3,
            device="cpu",
        )

        assert validator.num_segments == 1
        assert validator.durations == [16]
        assert validator.kernel_size == 32
        assert validator.stride_size == 4
        assert validator.pool_size == 8
        assert validator.integration_size == 2
        assert validator._injection_step == 8

    def test_steps_for_shift(self):
        mock = Mock()
        mock.current_duration = 20
        mock.kernel_length = 2
        mock.stride = 1.5
        mock.num_channels = 2

        steps = Validator.steps_for_shift(mock, 0)
        assert steps == 13

        steps = Validator.steps_for_shift(mock, 1)
        assert steps == 12

    def test_shift_background(self, background):
        mock = Mock()
        mock.sample_rate = 8
        mock.current_segment = background[0]
        mock.num_channels = 2

        background = Validator.shift_background(mock, 0)
        np.testing.assert_array_equal(background, mock.current_segment)

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

    def test_iter_shift(self, background, psd_estimator, whitener):
        mock = Mock()
        mock.current_segment = background[0]
        mock.sample_rate = 8
        mock.current_duration = 16
        mock.batch_size = 4
        mock.num_channels = 2
        mock.device = "cpu"
        mock.psd_estimator = psd_estimator
        mock.whitener = whitener

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

    def test_inject(self, waveforms):
        mock = Mock()
        mock._injection_step = 4
        mock._injection_idx = 0
        mock.kernel_size = 16
        mock.pad_size = 4
        mock.num_views = 3
        mock.snr_thresh = 4

        x = torch.arange(16, dtype=torch.float32)
        x = torch.stack([x, x])
        X = torch.stack([x + i for i in range(13)])
        X[:, 1] *= -1

        # waveforms will look like step functions in
        # the central 16 samples, with value equal to
        # the index (plus 1). The outer 8 samples on
        # each side will remain all 0s.
        waveforms = torch.zeros((3, 2, 32))
        waveforms[:, :, 8:-8] += torch.arange(3)[:, None, None] + 1
        mock.waveforms = waveforms

        def threshold_snrs(_, x, y):
            return x

        add_method(mock, threshold_snrs)

        injected, _ = Validator.inject(mock, X, X)
        injected = injected.numpy()
        assert injected.shape == (3 * 3, 2, 16)
        expected = np.arange(16)

        # fastest changing index along the batch dimension
        # is the waveform index. Slowest changing index
        # is the offset index (which view of the injection
        # we're taking)
        for i in range(3):
            for j in range(3):
                x = injected[i * 3 + j]
                y = expected + 5 * j + 1

                # for views outside the central one, either
                # the front end or the back end of the waveform
                # will be all zeros, so subtract off the value
                # the waveform adds (which is just its index)
                if i == 0:
                    y[-4:] -= j + 1
                elif i == 2:
                    y[:4] -= j + 1
                np.testing.assert_array_equal(x[0], y)

                # same goes for the second channel, but
                # accounting for the fact that it's negative
                # to begin with
                y = -expected - 3 * j + 1
                if i == 0:
                    y[-4:] -= j + 1
                elif i == 2:
                    y[:4] -= j + 1
                np.testing.assert_array_equal(x[1], y)

    def test_infer_shift(
        self, model, background, waveforms, psd_estimator, whitener
    ):
        return
        tracker = Mock()
        validator = Validator(
            tracker,
            background,
            waveforms,
            psd_estimator=psd_estimator,
            whitener=whitener,
            sample_rate=8,
            stride=0.5,
            injection_stride=4,
            snr_thresh=0,
            highpass=32,
            kernel_length=4,
            batch_size=8,
            pool_length=4,
            integration_length=1,
            livetime=32,
            shift=1,
            max_fpr=1e-3,
            device="cpu",
        )
        preds, inj_preds = validator.infer_shift(model, 0)
        preds = preds.numpy()

        # 14s worth of predictions, with 4s pooling,
        # and torch ditches the last 2s
        assert preds.shape == (3,)

        # NN predictions will be
        # [15, 19, 23, ...., 123, 127]
        # after integration
        # [7.5, 17, 21, ..., 125]
        # so first pooled output will be
        # 6 samples after the 17 above,
        # and then will be spaced evenly above that
        base = 17 + 4 * 6
        expected = base + np.arange(3) * 8 * 4
        np.testing.assert_array_equal(preds, expected)

        inj_preds = inj_preds.numpy()
        assert inj_preds.shape == (3,)

        # first waveform adds 1 to the last element
        # of the first kernel (15), second waveform
        # adds 2 to the last element of the kernel
        # 4s later (so 32 samples later), last waveform
        # adds 3 to the last element of the kernel
        # another 32 samples later
        expected = 15 + np.arange(3) * (4 * 8 + 1) + 1
        np.testing.assert_array_equal(inj_preds, expected)

    def test_call(self, model, background, waveforms, psd_estimator, whitener):
        tracker = Mock()
        tracker.log = lambda i, j: setattr(tracker, "metrics", j)

        validator = Validator(
            tracker,
            background,
            waveforms,
            psd_estimator=psd_estimator,
            whitener=whitener,
            sample_rate=8,
            stride=0.5,
            injection_stride=4,
            snr_thresh=0,
            highpass=32,
            kernel_length=2,
            batch_size=8,
            pool_length=4,
            integration_length=1,
            livetime=45,
            shift=1,
            max_fpr=1e-3,
            device="cpu",
        )

        values = (torch.zeros((3,)), torch.ones((3,)))
        with patch(
            "train.validation.Validator.infer_shift", return_value=values
        ) as mock:
            validator(model, 0.5)

        mock.assert_has_calls([call(model, i) for i in [1, -1, 2, -2]])
        assert tracker.metrics["train_loss"] == 0.5
        assert tracker.metrics["valid_auroc@1.0e-03"] == 1
