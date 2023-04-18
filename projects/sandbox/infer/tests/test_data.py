import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from infer import data

from bbhnet.analysis.ledger import injections


def ranger(sample_rate, N, step, shift):
    shift = int(sample_rate * shift)
    step = int(sample_rate * step)
    for i in range(N):
        x = step * i + np.arange(step)
        yield np.stack([x, x - shift]).astype("float32")


def test_shift_chunk():
    x = np.arange(10)
    x = np.stack([x, x - 1], axis=0)
    y, rems = data._shift_chunk(x, [0, 1], remainders=[None, None])
    assert y.shape == (2, 9)
    assert (y[0] == y[1]).all()

    assert len(rems) == 2
    assert len(rems[0]) == 1
    assert len(rems[1]) == 0

    x = x + 10
    y, rems = data._shift_chunk(x, [0, 1], rems)
    assert y.shape == (2, 10)
    assert (y[0] == np.arange(9, 19)).all()
    assert (y[0] == y[1]).all()

    assert len(rems) == 2
    assert len(rems[0]) == 1
    assert len(rems[1]) == 0


class TestSegmentIterator:
    @pytest.fixture(params=[128])
    def sample_rate(self, request):
        return request.param

    @pytest.fixture
    def response_set(self, sample_rate):
        duration = 1
        size = int(duration * sample_rate)
        N = 4
        kwargs = {}

        fields = injections.LigoResponseSet.__dataclass_fields__
        for name, attr in fields.items():
            if attr.metadata["kind"] == "parameter":
                kwargs[name] = np.zeros((N,))
            elif attr.metadata["kind"] == "waveform":
                wave = np.ones((N, size)) * (1 + np.arange(N)[:, None])
                kwargs[name] = wave.astype(np.float32)
        kwargs["gps_time"] = np.array([2.5, 5, 64, 91])
        kwargs["num_injections"] = 4
        kwargs["sample_rate"] = sample_rate
        kwargs["duration"] = duration
        return injections.LigoResponseSet(**kwargs)

    @pytest.fixture
    def validate_iteration(self, sample_rate, response_set):
        @patch(
            "bbhnet.analysis.ledger.injections.LigoResponseSet.read",
            return_value=response_set,
        )
        def f(shift, method, _):
            base_it = ranger(sample_rate, 10, 10, shift)

            def it():
                for i in base_it:
                    yield (i, None)

            it = data.SegmentIterator(
                it(), 0, 10, sample_rate, [0, shift], None
            )
            if method is not None:
                it = getattr(it, method)()

            outputs, outputs_inj = [], []
            for i, (x, x_inj) in enumerate(it):
                expected_length = int(10 * sample_rate)
                if not i:
                    expected_length -= int(shift * sample_rate)
                assert x.shape == x_inj.shape == (2, expected_length)
                outputs.append(x)
                outputs_inj.append(x_inj)
            assert i == 9

            output = np.concatenate(outputs, axis=1)
            output_inj = np.concatenate(outputs_inj, axis=1)

            stop = int(100 * sample_rate) - int(shift * sample_rate)
            assert output.shape[-1] == stop
            assert (output == np.arange(stop)).all()
            assert not (output_inj == np.arange(stop)).all()

            stop = int(2 * sample_rate)
            expected = np.arange(stop)
            diff = output_inj[:, :stop] - expected
            assert (output_inj[:, :stop] == expected).all(), diff[0]

            start = stop
            stop = int(3 * sample_rate)
            expected = np.arange(start, stop) + 1
            assert (output_inj[:, start:stop] == expected).all()

            start = stop
            stop = int(4.5 * sample_rate)
            expected = np.arange(start, stop)
            assert (output_inj[:, start:stop] == expected).all()

            start = stop
            stop = int(5.5 * sample_rate)
            expected = np.arange(start, stop) + 2
            assert (output_inj[:, start:stop] == expected).all()

            start = stop
            stop = int(63.5 * sample_rate)
            expected = np.arange(start, stop)
            assert (output_inj[:, start:stop] == expected).all()

        return f

    @patch(
        "bbhnet.analysis.ledger.injections.LigoResponseSet.read",
    )
    def test_init(self, sample_rate):
        obj = data.SegmentIterator(
            Mock(), 1234567890, 1234567900, sample_rate, [0, 5], None
        )
        # TODO: what else
        assert obj.duration == 10

    def test_shift_it(self, validate_iteration):
        validate_iteration(4, "_shift_it")

    def test_iter(self, validate_iteration):
        validate_iteration(5, None)


class TestSubsequence:
    def test_init(self):
        obj = data.Subsequence(128)
        assert obj.y.shape == (128,)
        assert not obj._idx
        assert not obj.done
        assert not obj.initialized

    def test_update(self):
        obj = data.Subsequence(128)
        for i in range(8):
            obj.update(np.ones((16, 1)) + i)
            assert obj.initialized
        assert obj.done
        for i in range(16):
            assert (obj.y[i::16] == np.arange(1, 9)).all()


class TestSequence:
    @pytest.fixture(params=[8])
    def sample_rate(self, request):
        return request.param

    @pytest.fixture(params=[4, 8, 32])
    def batch_size(self, request):
        return request.param

    def test_init(self, mock, sample_rate, batch_size):
        segment = Mock()
        segment.start = 0
        segment.end = 129
        segment.duration = 129

        injection_set = Mock()
        obj = data.Sequence(
            segment, injection_set, sample_rate, batch_size, 10
        )
        mock.assert_called_with(injection_set, start=0, end=129)
        assert not obj.done
        assert not obj.initialized

        if batch_size > sample_rate:
            assert obj.background.size == (128 * sample_rate)
            assert obj.num_steps == 128 / 4
        else:
            assert obj.background.size == (129 * sample_rate)
            if batch_size == sample_rate:
                assert obj.num_steps == 129
            else:
                assert obj.num_steps == 129 * 2

    @patch("infer.data.Sequence.inject", new=lambda _, x, __: x)
    def test_iter(self, _, sample_rate, batch_size):
        segment = Mock()
        segment.start = 0
        segment.end = 130
        segment.duration = 130
        segment.sample_rate = 128
        segment.__iter__ = lambda _: ranger(128, N=13, step=10, shift=0)

        sequence = data.Sequence(
            segment, "", sample_rate, batch_size, throughput=1200
        )

        dim = int(batch_size * 128 / sample_rate)
        for i, (x, y) in enumerate(sequence):
            assert x.shape == y.shape == (2, dim), i
            start, stop = i * dim, (i + 1) * dim
            assert (x == np.arange(start, stop)).all(), i
            sequence.background._idx = 1
            sequence.foreground._idx = 1
        assert (i + 1) == sequence.num_steps

        # test that our throughput is roughly correct
        segment.__iter__ = lambda _: ranger(128, N=26, step=10, shift=0)
        sequence = data.Sequence(
            segment, "", sample_rate, batch_size, throughput=130 * 8
        )
        start_time = time.time()
        for _ in sequence:
            sequence.foreground._idx = 1
            sequence.background._idx = 1
        end_time = time.time()
        duration = end_time - start_time
        assert 0.15 < duration < 0.3
