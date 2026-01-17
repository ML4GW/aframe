from utils.cosmology import (
    volume_element,
    get_astrophysical_volume,
    DEFAULT_COSMOLOGY,
)
import numpy as np


def test_zero_redshift_volume_element():
    assert np.isclose(volume_element(DEFAULT_COSMOLOGY, 0), 0)


def test_zero_redshift_astrophysical_volume():
    vol = get_astrophysical_volume(0, 0, DEFAULT_COSMOLOGY)
    assert np.isclose(vol, 0)


def test_zero_dec_range():
    vol = get_astrophysical_volume(0, 0.1, DEFAULT_COSMOLOGY, dec_range=(0, 0))
    assert np.isclose(vol, 0)


def test_full_sky_volume_double_half_sky():
    vol_full = get_astrophysical_volume(0, 0.1, DEFAULT_COSMOLOGY)
    vol_half = get_astrophysical_volume(
        0, 0.1, DEFAULT_COSMOLOGY, dec_range=(0, np.pi / 2)
    )
    assert np.isclose(vol_full, 2 * vol_half)
