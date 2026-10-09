from utils.streaming import StreamLayout, StreamOutputs


class TestStreamLayout:
    def test_partial_last_batch(self):
        stream_layout = StreamLayout(size=10, batch_size=2, stride=2)
        assert stream_layout.step_size == 4
        assert stream_layout.num_batches == 3
        assert stream_layout.remainder == 2
        assert stream_layout.num_pad == 2

        assert stream_layout.num_pad_outputs == 1
        assert stream_layout.batch_bounds(0) == (0, 4, False)
        assert stream_layout.batch_bounds(1) == (4, 8, False)
        assert stream_layout.batch_bounds(2) == (8, 10, True)

    def test_full_last_batch(self):
        stream_layout = StreamLayout(size=8, batch_size=2, stride=2)
        assert stream_layout.num_batches == 2
        assert stream_layout.remainder == 0
        assert stream_layout.num_pad == 0
        assert stream_layout.num_pad_outputs == 0
        assert stream_layout.batch_bounds(1) == (4, 8, True)


class TestStreamOutputs:
    def test_sizes_and_times(self):
        stream_outputs = StreamOutputs(
            inference_sampling_rate=4.0,
            fduration=1.0,
            integration_window_length=1.5,
            psd_length=1.0,
        )
        assert stream_outputs.burn_in_size == 4
        assert stream_outputs.integration_size == 7
        assert stream_outputs.lag == 2.0

        assert stream_outputs.timestamp(1000.0, 2) == 998.75
        assert stream_outputs.index(998.75, 1000.0) == 2
