from math import isclose

import numpy as np
import pytest
from infer.callback import Callback, ExistingSequence


@pytest.fixture
def inference_sampling_rate():
    return 256


@pytest.mark.parametrize(
    "integration_window_length,cluster_window_length", [(1, 8), (2, 8), (1, 4)]
)
class TestCallback:
    @pytest.fixture
    def callback(
        self,
        integration_window_length,
        cluster_window_length,
        inference_sampling_rate,
    ):
        return Callback(
            id=0,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=16,
            integration_window_length=integration_window_length,
            cluster_window_length=cluster_window_length,
            fduration=1.0,
            psd_length=16,
        )

    def test_initialize(self, callback):
        assert callback.start is None
        start, stop = 0, 1
        callback.initialize(start, stop)

        with pytest.raises(ExistingSequence):
            callback.initialize(start, stop)

    def test_integrate(self, callback):
        y = np.arange(callback.inference_sampling_rate * 10) + 1
        integrated = callback.integrate(y)
        assert len(integrated) == len(y)
        window_size = int(
            callback.integration_window_length
            * callback.inference_sampling_rate
        )
        for i, value in enumerate(integrated):

            if i < window_size:
                expected = sum(range(i + 2)) / window_size
            else:
                expected = (i + 3 + i - window_size) / 2

        assert isclose(value, expected, rel_tol=1e-9)
