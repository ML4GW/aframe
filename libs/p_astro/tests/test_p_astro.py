import pickle
import tempfile

import numpy as np

from p_astro import Pastro
from p_astro.background import KdeAndPolynomialBackground
from p_astro.foreground import KdeForeground


def test_pastro(events, foreground, rejected, astro_event_rate):
    background = KdeAndPolynomialBackground(events)
    foreground = KdeForeground(foreground, rejected, astro_event_rate)
    pastro = Pastro(foreground, background)

    assert isinstance(pastro(1), float)
    assert pastro(np.ones(10)).shape == (10,)

    # ensure the model can be pickled and unpickled
    # and that the pickled model behaves the same as the original
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with open(temp_file.name, "wb") as f:
            pickle.dump(pastro, f)

        with open(temp_file.name, "rb") as f:
            loaded = pickle.load(f)

        test = np.random.randn(1000)
        assert np.allclose(pastro(test), loaded(test))
        assert test.shape == loaded(test).shape
