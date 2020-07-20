"""
test_prior.py : test the model priors 
"""
import numpy as np
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.set_parameters import Parameters


def test_prior_catalog():
    params = Parameters()
    prior = PriorCatalog(
        params,
        "data/dr12q/processed/catalog.mat",
        "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    )

    # 94892842 2.0969 20.0292
    ind = prior.thing_ids == 94892842
    assert np.all(prior.z_dlas[ind] == 2.0969)
    assert np.all(prior.log_nhis[ind] == 20.0292)

    # P(DLA | zQSO) prior would saturated at around 0.1
    zQSO = 5
    div = lambda M, N: M / N
    p_dla_z = div(*prior.less_ind(zQSO))
    assert 0.09 < p_dla_z < 0.11
