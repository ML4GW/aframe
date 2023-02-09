import math
from unittest.mock import Mock, patch

import astropy.units as u
import bilby
import pytest

from bbhnet.analysis.sensitivity import (
    SensitiveVolumeCalculator,
    calculate_astrophysical_volume,
)


@pytest.fixture()
def prior():
    prior = bilby.core.prior.PriorDict(
        dict(
            m1=bilby.core.prior.Uniform(0, 1, "m1", "Msol"),
            m2=bilby.core.prior.Uniform(0, 1, "m2", "Msol"),
            redshift=bilby.core.prior.Uniform(100, 1000, "redshift"),
            dec=bilby.core.prior.analytical.Cosine(name="dec"),
            ra=bilby.core.prior.Uniform(
                minimum=0, maximum=2 * math.pi, name="ra"
            ),
        )
    )
    return prior


def test_sensitive_volume(prior):
    sensitive_volume_calculator = SensitiveVolumeCalculator(
        source=prior,
        recovered_parameters=prior.sample(100),
        n_injections=100,
    )

    # calculating weights without target
    # should produce weights of 1s
    weights = sensitive_volume_calculator.weights()
    assert all(weights == 1)

    # calculating weights with source
    # as target should alo produce weights of 1s
    weights = sensitive_volume_calculator.weights(target=prior)
    assert all(weights == 1)

    # all weights are 1, so setting the volume to 1
    # should just return the V0
    sensitive_volume_calculator.volume = 1 * u.Mpc**3
    (
        sensitive_volume,
        _,
        _,
    ) = sensitive_volume_calculator.calculate_sensitive_volume()
    assert sensitive_volume == sensitive_volume_calculator.volume.value

    # TODO: add test for calculating vt with non-trivial target


def test_calculate_astrophysical_volume():
    # create mock cosmology where dv/dz
    # cancels out the additional 1 + z term
    # so the integrand is just 1
    cosmology = Mock()
    cosmology.differential_comoving_volume = lambda x: (1 + x) * u.Mpc

    dl_min, dl_max = 0, 1
    with patch(
        "bbhnet.analysis.sensitivity.cosmo.z_at_value",
        return_value=[dl_min, dl_max],
    ):
        volume = calculate_astrophysical_volume(
            dl_min, dl_max, cosmology=cosmology
        )
        # expected answer is 4 pi since the integrand
        # is 1 and the volume is 1 Mpc^3
        assert volume.value == 4 * math.pi
