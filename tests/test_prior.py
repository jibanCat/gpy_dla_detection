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

# generate a DLA instance from DLAGP class
from .test_model import prepare_dla_model, prepare_subdla_model


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


def test_subdla_and_null_model_priors():
    dla_gp = prepare_dla_model(plate=5309, mjd=55929, fiber_id=362, z_qso=3.166)
    subdla_gp = prepare_subdla_model(plate=5309, mjd=55929, fiber_id=362, z_qso=3.166)

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
        [-2.53774598, -4.97413739, -7.40285925, -9.74851888]
    )
    print(catalog_log_priors, log_priors_4_dla)
    assert np.all(np.abs(log_priors_4_dla - catalog_log_priors) < 1e-4)

    # check if the subDLA model priors make sense
    log_priors_1_subdla = subdla_gp.log_priors(subdla_gp.z_qso, 1)[0]
    assert 0 <= np.exp(log_priors_1_subdla) <= 1
    assert (
        np.abs(
            np.exp(log_priors_1_dla)
            - subdla_gp.dla_samples._Z_dla
            / subdla_gp.dla_samples._Z_lls
            * np.exp(log_priors_1_subdla)
        )
        < 1e-2
    )

    # check if the null model is the same as the catalogue value
    log_prior_no_dla = np.log(
        1 - np.exp(log_priors_1_subdla) - np.exp(logsumexp(log_priors_4_dla))
    )
    print(log_prior_no_dla)
    catalog_log_prior_no_dla = -0.14987896

    print("log p( no DLA | z_QSO ) : {:.5g}; MATLAB value: {:.5g}".format(
            log_prior_no_dla, catalog_log_prior_no_dla))
    assert np.abs(log_prior_no_dla - catalog_log_prior_no_dla) < 1e-3

    # test 2
    dla_gp = prepare_dla_model(3816, 55272, 76, z_qso=3.68457627)
    subdla_gp = prepare_subdla_model(3816, 55272, 76, z_qso=3.68457627)

    # test model priors
    log_priors_1_dla = dla_gp.log_priors(dla_gp.z_qso, 1)
    log_priors_2_dla = dla_gp.log_priors(dla_gp.z_qso, 2)
    log_priors_4_dla = dla_gp.log_priors(dla_gp.z_qso, 4)

    # make sure they sum to the same value, since prior for
    # at least one DLA should equal to prior for exactly
    # one DLA + at least two DLAs.
    assert np.abs(logsumexp(log_priors_1_dla) - logsumexp(log_priors_2_dla)) < 5e-2
    assert np.abs(logsumexp(log_priors_2_dla) - logsumexp(log_priors_4_dla)) < 5e-2 # TODO: the bug of MATLAB needs to be fix in the future

    # the prior values in the Ho-Bird-Garnett catalog
    catalog_log_priors = np.array(
        [-2.40603132, -4.69090863, -6.96539755, -9.14424546]
    )
    print(catalog_log_priors, log_priors_4_dla)
    assert np.all(np.abs(log_priors_4_dla - catalog_log_priors) < 1e-3)

    # check if the subDLA model priors make sense
    log_priors_1_subdla = subdla_gp.log_priors(subdla_gp.z_qso, 1)[0]
    assert 0 <= np.exp(log_priors_1_subdla) <= 1
    assert (
        np.abs(
            np.exp(log_priors_1_dla)
            - subdla_gp.dla_samples._Z_dla
            / subdla_gp.dla_samples._Z_lls
            * np.exp(log_priors_1_subdla)
        )
        < 1e-2
    )

    # check if the null model is the same as the catalogue value
    log_prior_no_dla = np.log(
        1 - np.exp(log_priors_1_subdla) - np.exp(logsumexp(log_priors_4_dla))
    )
    catalog_log_prior_no_dla = -0.17660122

    print("log p( no DLA | z_QSO ) : {:.5g}; MATLAB value: {:.5g}".format(
            log_prior_no_dla, catalog_log_prior_no_dla))
    assert np.abs(log_prior_no_dla - catalog_log_prior_no_dla) < 5e-3
