import astropy.cosmology as cosmo


def planck():
    """
    Return an `astropy` cosmology defined by the
    Planck Collaboration 2015 paper
    """
    return cosmo.Planck15
