from bbhnet.priors.priors import PriorDict


def test_prior_dict():
    prior = PriorDict(source_frame=True)
    prior["mass_1"] = 1
    prior["mass_2"] = 2
    prior["redshift"] = 3

    samples = prior.sample(10, source_frame=True)
    assert all(samples["mass_1"] == 1)
    assert all(samples["mass_2"] == 2)

    samples = prior.sample(10, source_frame=False)
    assert all(samples["mass_1"] == 1 * (1 + 3))
    assert all(samples["mass_2"] == 2 * (1 + 3))

    prior = PriorDict(source_frame=False)
    prior["mass_1"] = 1
    prior["mass_2"] = 2
    prior["redshift"] = 3

    samples = prior.sample(10, source_frame=False)
    assert all(samples["mass_1"] == 1)
    assert all(samples["mass_2"] == 2)

    samples = prior.sample(10, source_frame=True)
    assert all(samples["mass_1"] == 1 / (1 + 3))
    assert all(samples["mass_2"] == 2 / (1 + 3))
