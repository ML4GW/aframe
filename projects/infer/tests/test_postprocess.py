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
    assert postprocessor.t0 == 9.0
