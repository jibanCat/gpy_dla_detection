'''
Test the Cython functions
'''
import os
import time
import numpy as np
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.read_spec import read_spec, retrieve_raw_spec
from gpy_dla_detection.dla_samples import DLASamplesMAT

from fast_dla_log_model_evidence import this_dla_gp, sample_log_likelihood_k_dlas, log_model_evidences

def test_this_dla_profile():
    # test 1
    filename = "spec-5309-55929-0362.fits"

    if not os.path.exists(filename):
        retrieve_raw_spec(5309, 55929, 362)  # the spectrum at paper

    z_qso = 3.166

    param = Parameters()

    # prepare these files by running the MATLAB scripts until build_catalog.m
    prior = PriorCatalog(
        param,
        "data/dr12q/processed/catalog.mat",
        "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    )
    dla_samples = DLASamplesMAT(
        param, prior, "data/dr12q/processed/dla_samples_a03.mat"
    )

    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
    rest_wavelengths = param.emitted_wavelengths(wavelengths, z_qso)

    # DLA GP Model
    dla_gp = DLAGPMAT(
        param,
        prior,
        dla_samples,
        3000.0,
        "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
    )
    dla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # These are the MAPs from the paper
    z_dlas = np.array([2.52182382, 3.03175723])
    nhis = 10 ** np.array([20.63417494, 22.28420156])

    # prepare arrays
    padded_wavelengths = dla_gp.padded_wavelengths
    this_mu = dla_gp.this_mu
    this_M = dla_gp.this_M
    this_omega2 = dla_gp.this_omega2
    mask_index = np.where(~dla_gp.pixel_mask[dla_gp.ind])[0]
    num_lines = dla_gp.params.num_lines

    dla_mu, dla_M, dla_omega2 = this_dla_gp(
        z_dlas, nhis, padded_wavelengths, this_mu, this_M, this_omega2, mask_index, num_lines)

    dla_mu = np.asarray(dla_mu)
    dla_M = np.asarray(dla_M)
    dla_omega2 = np.asarray(dla_omega2)

    dla_mu2, dla_M2, dla_omega22 = dla_gp.this_dla_gp(z_dlas, nhis)

    assert np.all(np.abs(dla_mu - dla_mu2) < 1e-4)
    assert np.all(np.abs(dla_M - dla_M2) < 1e-4)
    assert np.all(np.abs(dla_omega2 - dla_omega22) < 1e-4)

    # calculate log sample likelihood
    y = dla_gp.y.astype(np.double)
    v = dla_gp.v.astype(np.double)

    # naive timeit
    tic = time.time()
    for i in range(100):
        sample_log_likelihood = sample_log_likelihood_k_dlas(
            y, v, z_dlas, nhis, padded_wavelengths, this_mu, this_M, this_omega2, mask_index, num_lines
        )
        time_spent = time.time() - tic
        tic = time.time()
    print("Spent {:.5g} with Cython".format(time_spent / 50))

    tic = time.time()
    for i in range(100):
        sample_log_likelihood2 = dla_gp.sample_log_likelihood_k_dlas(z_dlas, nhis)
        time_spent = time.time() - tic
        tic = time.time()
    print("Spent {:.5g} with Native python".format(time_spent / 50))

    assert np.abs(sample_log_likelihood - sample_log_likelihood2) < 1e-4

def test_log_model_evidences():
    # test 1
    filename = "spec-5309-55929-0362.fits"

    if not os.path.exists(filename):
        retrieve_raw_spec(5309, 55929, 362)  # the spectrum at paper

    z_qso = 3.166

    param = Parameters()

    # prepare these files by running the MATLAB scripts until build_catalog.m
    prior = PriorCatalog(
        param,
        "data/dr12q/processed/catalog.mat",
        "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    )
    dla_samples = DLASamplesMAT(
        param, prior, "data/dr12q/processed/dla_samples_a03.mat"
    )

    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
    rest_wavelengths = param.emitted_wavelengths(wavelengths, z_qso)

    # DLA GP Model
    dla_gp = DLAGPMAT(
        param,
        prior,
        dla_samples,
        3000.0,
        "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
    )
    dla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # prepare arrays
    # model inputs
    this_mu = dla_gp.this_mu
    this_M = dla_gp.this_M
    this_omega2 = dla_gp.this_omega2
    num_lines = dla_gp.params.num_lines

    # data input
    y = dla_gp.y.astype(np.double)
    v = dla_gp.v.astype(np.double)
    mask_index = np.where(~dla_gp.pixel_mask[dla_gp.ind])[0]
    padded_wavelengths = dla_gp.padded_wavelengths

    # multi-DLA min separation
    min_z_separation = dla_gp.min_z_separation
    
    # DLA sampling part
    sample_z_dlas = dla_gp.dla_samples.sample_z_dlas(dla_gp.this_wavelengths, dla_gp.z_qso)
    nhi_samples = dla_gp.dla_samples.nhi_samples
    num_dla_samples = dla_gp.dla_samples.num_dla_samples

    tic = time.time()

    # test the Cython sample function
    max_dlas = 4
    log_likelihoods_dla, sample_log_likelihoods = log_model_evidences(
        y,
        v,
        max_dlas,
        num_dla_samples,
        sample_z_dlas,
        nhi_samples,
        padded_wavelengths,
        this_mu,
        this_M,
        this_omega2,
        mask_index, # np.where(mask_ind)[0]
        num_lines,
        min_z_separation
    )

    toc = time.time()
    # very time consuming: ~ 4 mins for a single spectrum without parallelized.
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    log_likelihoods_dla = np.asarray(log_likelihoods_dla)

    # log likelihood results from the catalog
    catalog_log_likelihoods_dla = np.array(
        [-688.91647288, -633.00070813, -634.08569242, -640.77120558]
    )

    for i in range(max_dlas):
        print(
            "log p(  D  | z_QSO, DLA{} ) : {:.5g}; MATLAB value: {:.5g}".format(
                i + 1, log_likelihoods_dla[i], catalog_log_likelihoods_dla[i]
            )
        )

    # the accuracy down to 2.5 in log scale, this needs to be investigated.
    assert np.all(np.abs(catalog_log_likelihoods_dla - log_likelihoods_dla) < 2.5)
