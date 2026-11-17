import pickle
import tempfile

import numpy as np
import pytest
from ledger.events import RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from p_astro.foreground import KdeForeground
from scipy.integrate import quad
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


def known_answer_ledgers(prior, z0):
    # draws from the prior, with every source below z0 accepted
    draws = prior.sample(200_000)
    accepted = draws["redshift"] < z0
    foreground = make_ledger(
        RecoveredInjectionSet,
        redshift=draws["redshift"][accepted],
        dec=draws["dec"][accepted],
        detection_statistic=np.random.default_rng().standard_normal(
            accepted.sum()
        ),
        num_injections=int(accepted.sum()),
    )
    rejected = make_ledger(
        InjectionParameterSet,
        redshift=draws["redshift"][~accepted],
        dec=draws["dec"][~accepted],
    )
    return foreground, rejected


def test_foreground_rate_follows_target_redshift(source_prior):
    # draws uniform in comoving volume: the total rate is the rate
    # density times the time-dilated volume below z0
    rate, z0 = 19.0, 1.0
    foreground, rejected = known_answer_ledgers(source_prior, z0)

    model = KdeForeground(foreground, rejected, rate, source_prior)

    expected = rate * get_astrophysical_volume(0, z0, DEFAULT_COSMOLOGY) / 1e9
    assert model.scale_factor == pytest.approx(expected, rel=0.03)


def test_foreground_rate_follows_rate_evolution(source_prior):
    # a rate of 30.4 at z = 0.2 that rises as (1 + z)^2.5: the total
    # rate is the integral of rate times time-dilated volume below z0
    rate, rate_index, rate_redshift, z0 = 30.4, 2.5, 0.2, 1.0
    foreground, rejected = known_answer_ledgers(source_prior, z0)

    model = KdeForeground(
        foreground,
        rejected,
        rate,
        source_prior,
        rate_index=rate_index,
        rate_redshift=rate_redshift,
    )

    def rate_per_redshift(z):
        volume = DEFAULT_COSMOLOGY.differential_comoving_volume(z).value
        evolution = ((1 + z) / (1 + rate_redshift)) ** rate_index
        return rate * 4 * np.pi * volume / (1 + z) * evolution

    expected = quad(rate_per_redshift, 0, z0)[0] / 1e9
    assert model.scale_factor == pytest.approx(expected, rel=0.03)
