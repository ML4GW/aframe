import numpy as np
from infer.postprocess import Postprocessor


def test_postprocessor():
    postprocessor = Postprocessor(
        t0=0.0,
        shifts=[0.1, 0.2, 0.3],
        psd_length=10.0,
        fduration=1.0,
        inference_sampling_rate=100.0,
        integration_window_length=0.5,
        cluster_window_length=0.2,
    )
    assert postprocessor.t0 == 9.01


def test_postprocessor_times():
    postprocessor = Postprocessor(
        t0=100.0,
        shifts=[0.0, 1.0],
        psd_length=2.0,
        fduration=1.0,
        inference_sampling_rate=4.0,
        integration_window_length=1.5,
        cluster_window_length=2.0,
    )
    y = np.zeros(200)
    y[40] = 10.0
    events = postprocessor(y)
    loudest = events.detection_time[np.argmax(events.detection_statistic)]
    # available at 100 + 41 / 4, timestamped 0.5 + 1.5 earlier
    assert loudest == 108.25
