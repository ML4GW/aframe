import math
from unittest.mock import Mock, patch

import astropy.units as u
import bilby
import pytest

from bbhnet.analysis.sensitivity import (
    SensitiveVolumeCalculator,
    calculate_astrophysical_volume,
)
from bbhnet.priors.utils import transpose


@pytest.fixture()
def prior():
    def prior_func(cosmology=None):
        prior = bilby.core.prior.PriorDict(
            dict(
                mass_1=bilby.core.prior.Uniform(0, 1, "mass_1", "Msol"),
                mass_2=bilby.core.prior.Uniform(0, 1, "mass_2", "Msol"),
                redshift=bilby.core.prior.Uniform(100, 1000, "redshift"),
                dec=bilby.core.prior.analytical.Cosine(name="dec"),
                ra=bilby.core.prior.Uniform(
                    minimum=0, maximum=2 * math.pi, name="ra"
                ),
            )
        )
        detector_frame_prior = True
        return prior, detector_frame_prior

    return prior_func


def test_sensitive_volume(prior):
    sensitive_volume_calculator = SensitiveVolumeCalculator(prior)
    prior, _ = prior()
    n_samples, n_injections = 100, 200
    recovered_parameters = prior.sample(n_samples)
    recovered_parameters = transpose(recovered_parameters)

    # calculating weights without target
    # should produce weights of 1s
    weights = sensitive_volume_calculator.weights(recovered_parameters, prior)
    assert all(weights == 1)

    # all weights are 1, so setting the volume to 1
    # should just return the V0
    sensitive_volume_calculator.volume = 1 * u.Mpc**3
    (
        sensitive_volume,
        std,
        n_eff,
    ) = sensitive_volume_calculator(recovered_parameters, n_injections)
    assert sensitive_volume == sensitive_volume_calculator.volume.value * (
        n_samples / n_injections
    )
    assert (
        n_eff == (n_samples / n_injections) ** 2 / std**2
    )  # this works since volume = 1

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
        assert volume == 4 * math.pi
