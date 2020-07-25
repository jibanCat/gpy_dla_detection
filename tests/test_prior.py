"""
test_prior.py : test the model priors 
"""
import os
import numpy as np
from scipy.special import logsumexp
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.read_spec import retrieve_raw_spec, read_spec
from gpy_dla_detection.dla_samples import DLASamplesMAT

from .test_model import prepare_dla_model  # generate a DLA instance from DLAGP class


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


def test_dla_model_priors():
    dla_gp = prepare_dla_model(plate=5309, mjd=55929, fiber_id=362)

    # test model priors
    log_priors_1_dla = dla_gp.log_priors(dla_gp.z_qso, 1)
    log_priors_2_dla = dla_gp.log_priors(dla_gp.z_qso, 2)
    log_priors_4_dla = dla_gp.log_priors(dla_gp.z_qso, 4)

    # make sure they sum to the same value, since prior for
    # at least one DLA should equal to prior for exactly
    # one DLA + at least two DLAs.
    assert np.abs(logsumexp(log_priors_1_dla) - logsumexp(log_priors_2_dla)) < 1e-2
    assert np.abs(logsumexp(log_priors_2_dla) - logsumexp(log_priors_4_dla)) < 1e-2

    # the prior values in the Ho-Bird-Garnett catalog
    catalog_log_priors = np.array(
        [[-2.53774598, -4.97413739, -7.40285925, -9.74851888]]
    )
    assert np.all(np.abs(log_priors_4_dla - catalog_log_priors) < 1e-4)
