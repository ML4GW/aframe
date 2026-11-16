import pickle
import tempfile

import numpy as np
import pytest
from bilby.core.prior import Cosine, PriorDict
from bilby.gw.prior import UniformComovingVolume
from ledger.events import RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from p_astro.foreground import KdeForeground
from utils.cosmology import DEFAULT_COSMOLOGY, get_astrophysical_volume


def test_foreground(foreground, rejected, astro_event_rate, source_prior):
    # test that the model can be called
    # on floats and numpy arrays
    model = KdeForeground(foreground, rejected, astro_event_rate, source_prior)

    assert isinstance(model(1), float)
    assert model(np.ones(10)).shape == (10,)

    # ensure the model can be pickled and unpickled
    # and that the pickled model behaves the same as the original
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with open(temp_file.name, "wb") as f:
            pickle.dump(model, f)

        with open(temp_file.name, "rb") as f:
            loaded = pickle.load(f)

        test = np.random.default_rng().standard_normal(1000)
        assert np.allclose(model(test), loaded(test))
        assert test.shape == loaded(test).shape


def make_ledger(cls, **fields):
    n = len(fields["redshift"])
    data = dict.fromkeys(cls.__dataclass_fields__, np.zeros(n))
    return cls(**{**data, **fields})


def test_foreground_rate_follows_target_redshift():
    # draws uniform in comoving volume, every source below z0 found:
    # the total rate is the rate density times the time-dilated
    # volume below z0
    rate, z0 = 19.0, 1.0
    prior = PriorDict(
        {
            "redshift": UniformComovingVolume(
                0, 2, name="redshift", cosmology=DEFAULT_COSMOLOGY
            ),
            "dec": Cosine(),
        }
    )
    draws = prior.sample(200_000)
    found = draws["redshift"] < z0
    foreground = make_ledger(
        RecoveredInjectionSet,
        redshift=draws["redshift"][found],
        dec=draws["dec"][found],
        detection_statistic=np.random.default_rng().standard_normal(
            found.sum()
        ),
        num_injections=int(found.sum()),
    )
    rejected = make_ledger(
        InjectionParameterSet,
        redshift=draws["redshift"][~found],
        dec=draws["dec"][~found],
    )

    model = KdeForeground(foreground, rejected, rate, prior)

    expected = rate * get_astrophysical_volume(0, z0, DEFAULT_COSMOLOGY) / 1e9
    assert model.scale_factor == pytest.approx(expected, rel=0.03)
