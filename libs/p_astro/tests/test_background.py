import pickle
import tempfile

import numpy as np

from p_astro.background import KdeAndPolynomialBackground


def test_background(events):
    # test that the model can be called
    # on floats and numpy arrays
    model = KdeAndPolynomialBackground(events)

    assert isinstance(model(1), float)
    assert model(np.ones(10)).shape == (10,)

    # ensure the model can be pickled and unpickled
    # and that the pickled model behaves the same as the original
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with open(temp_file.name, "wb") as f:
            pickle.dump(model, f)

        with open(temp_file.name, "rb") as f:
            loaded = pickle.load(f)

        test = np.random.randn(1000)
        assert np.allclose(model(test), loaded(test))
        assert test.shape == loaded(test).shape
