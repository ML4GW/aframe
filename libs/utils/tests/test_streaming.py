from utils.streaming import StreamLayout, StreamTiming


class TestStreamLayout:
    def test_partial_last_batch(self):
        layout = StreamLayout(size=10, batch_size=2, stride=2)
        assert layout.step_size == 4
        assert layout.num_batches == 3
        assert layout.remainder == 2
        assert layout.num_pad == 2

        assert layout.num_pad_outputs == 1
        assert layout.batch_bounds(0) == (0, 4, False)
        assert layout.batch_bounds(1) == (4, 8, False)
        assert layout.batch_bounds(2) == (8, 10, True)

    def test_full_last_batch(self):
        layout = StreamLayout(size=8, batch_size=2, stride=2)
        assert layout.num_batches == 2
        assert layout.remainder == 0
        assert layout.num_pad == 0
        assert layout.num_pad_outputs == 0
        assert layout.batch_bounds(1) == (4, 8, True)


class TestStreamTiming:
    def test_sizes_and_times(self):
        timing = StreamTiming(
            inference_sampling_rate=4.0,
            fduration=1.0,
            integration_window_length=1.5,
            psd_length=1.0,
        )
        assert timing.burn_in_size == 4
        assert timing.integration_size == 7
        assert timing.lag == 2.0

        assert timing.output_time(1000.0, 2) == 998.75
        assert timing.output_index(998.75, 1000.0) == 2
