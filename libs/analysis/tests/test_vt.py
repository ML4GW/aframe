import math
from unittest.mock import Mock, patch

import astropy.units as u
import bilby
import pytest

from bbhnet.analysis.vt import (
    YEARS_PER_SECOND,
    VolumeTimeIntegral,
    calculate_astrophysical_volume,
)


@pytest.fixture()
def prior():
    prior = bilby.core.prior.PriorDict(
        dict(
            m1=bilby.core.prior.Uniform(0, 1, "m1", "Msol"),
            m2=bilby.core.prior.Uniform(0, 1, "m2", "Msol"),
            luminosity_distance=bilby.core.prior.Uniform(
                100, 1000, "luminosity_distance", "Mpc"
            ),
            dec=bilby.core.prior.analytical.Cosine(name="dec"),
            ra=bilby.core.prior.Uniform(
                minimum=0, maximum=2 * math.pi, name="ra"
            ),
        )
    )
    return prior


def test_volume_time_integral(prior):
    vt = VolumeTimeIntegral(
        source=prior,
        recovered_parameters=prior.sample(100),
        n_injections=100,
        livetime=100,
    )

    # calculating weights without target
    # should produce weights of 1s
    weights = vt.weights()
    assert all(weights == 1)

    # calculating weights with source
    # as target should alo produce weights of 1s
    weights = vt.weights(target=prior)
    assert all(weights == 1)

    # all weights are 1, so setting the volume to 1
    # should just return the livetime for the vt calculation
    vt.volume = 1 * u.Mpc**3
    volume_time, _, _ = vt.calculate_vt()
    assert volume_time == (100 * YEARS_PER_SECOND)

    # TODO: add test for calculating vt with non-trivial target


def test_calculate_astrophysical_volume():
    # create mock cosmology where dv/dz
    # cancels out the additional 1 + z term
    # so the integrand is just 1
    cosmology = Mock()
    cosmology.differential_comoving_volume = lambda x: (1 + x) * u.Mpc

    dl_min, dl_max = 0, 1
    with patch(
        "bbhnet.analysis.vt.cosmo.z_at_value", return_value=[dl_min, dl_max]
    ):
        volume = calculate_astrophysical_volume(
            dl_min, dl_max, cosmology=cosmology
        )
        # expected answer is 4 pi since the integrand
        # is 1 and the volume is 1 Mpc^3
        assert volume.value == 4 * math.pi
