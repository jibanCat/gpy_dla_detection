"""
test_model.py : test the functions related to GP model
"""
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from gpy_dla_detection.effective_optical_depth import effective_optical_depth
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.null_gp import NullGPMAT, NullGP
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.read_spec import read_spec, retrieve_raw_spec
from gpy_dla_detection.dla_samples import DLASamplesMAT


def test_effective_optical_depth():
    z_qso = 4
    rest_wavelengths = np.linspace(911, 1216, 500)
    wavelengths = Parameters.observed_wavelengths(rest_wavelengths, z_qso)

    total_optical_depth = effective_optical_depth(
        wavelengths, 3.65, 0.0023, z_qso, 31, True
    )

    assert 0 < np.exp(-total_optical_depth.sum(axis=1)).min() < 1
    assert 0 < np.exp(-total_optical_depth.sum(axis=1)).max() < 1

    total_optical_depth_2 = effective_optical_depth(
        wavelengths, 3.65, 0.0023, z_qso, 1, True
    )

    assert np.mean(np.exp(-total_optical_depth.sum(axis=1))) < np.mean(
        np.exp(-total_optical_depth_2.sum(axis=1))
    )

    total_optical_depth_3 = effective_optical_depth(
        wavelengths, 3.65, 0.0023, 2.2, 31, True
    )

    assert np.mean(np.exp(-total_optical_depth.sum(axis=1))) < np.mean(
        np.exp(-total_optical_depth_3.sum(axis=1))
    )


def test_log_mvnpdf():
    y = np.array([1, 2])
    mu = np.array([1, 2])
    M = np.array([[2, 3, 1], [1, 2, 4]])
    d = np.eye(2) * 2

    rv = multivariate_normal(mu, np.matmul(M, M.T) + d)

    log_p = NullGP.log_mvnpdf_low_rank(y, mu, M, np.ones(2) * 2)

    assert np.abs(log_p - np.log(rv.pdf(y))) < 1e-4

    y = np.array([2, 3])

    log_p = NullGP.log_mvnpdf_low_rank(y, mu, M, np.ones(2) * 2)
    assert np.abs(log_p - np.log(rv.pdf(y))) < 1e-4

    y = np.array([100, 100])

    log_p = NullGP.log_mvnpdf_low_rank(y, mu, M, np.ones(2) * 2)
    assert np.abs(log_p - np.log(rv.pdf(y))) < 1e-4


def test_log_likelihood_no_dla():
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

    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
    rest_wavelengths = param.emitted_wavelengths(wavelengths, z_qso)

    gp = NullGPMAT(
        param,
        prior,
        "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
    )

    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    log_likelihood_no_dla = gp.log_model_evidence()
    print("log p(  D  | z_QSO, no DLA ) : {:.5g}".format(log_likelihood_no_dla))

    assert (
        np.abs(log_likelihood_no_dla - (-889.04809017)) < 1
    )  # there is some numerical difference

    plt.figure(figsize=(16, 5))
    plt.plot(gp.x, gp.y, label="observed flux")
    plt.plot(gp.rest_wavelengths, gp.mu, label="null GP")
    plt.plot(gp.x, gp.this_mu, label="interpolated null GP")
    plt.xlabel("rest wavelengths")
    plt.ylabel("normalised flux")
    plt.legend()
    plt.savefig("test1.pdf", format="pdf", dpi=300)
    plt.clf()
    plt.close()

    # test 2
    filename = "spec-3816-55272-0076.fits"
    z_qso = 3.68457627

    if not os.path.exists(filename):
        retrieve_raw_spec(3816, 55272, 76)  # the spectrum at paper

    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
    rest_wavelengths = param.emitted_wavelengths(wavelengths, z_qso)

    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    log_likelihood_no_dla = gp.log_model_evidence()
    print("log p(  D  | z_QSO, no DLA ) : {:.5g}".format(log_likelihood_no_dla))

    assert np.abs(log_likelihood_no_dla - (-734.3727266)) < 1

    plt.figure(figsize=(16, 5))
    plt.plot(gp.x, gp.y, label="observed flux")
    plt.plot(gp.rest_wavelengths, gp.mu, label="null GP")
    plt.plot(gp.x, gp.this_mu, label="interpolated null GP")
    plt.xlabel("rest wavelengths")
    plt.ylabel("normalised flux")
    plt.legend()
    plt.savefig("test2.pdf", format="pdf", dpi=300)
    plt.clf()
    plt.close()


def test_dla_model():
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

    sample_log_likelihood_dla = dla_gp.sample_log_likelihood_k_dlas(z_dlas, nhis)
    print(
        "log p(  D  | z_QSO, zdlas, nhis ) : {:.5g}".format(sample_log_likelihood_dla)
    )

    # Build a Null model
    gp = NullGPMAT(
        param,
        prior,
        "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
    )
    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    log_likelihood_no_dla = gp.log_model_evidence()
    print("log p(  D  | z_QSO, no DLA ) : {:.5g}".format(log_likelihood_no_dla))

    assert sample_log_likelihood_dla > log_likelihood_no_dla


def test_dla_model_evidences():
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

    tic = time.time()

    max_dlas = 4
    log_likelihoods_dla = dla_gp.log_model_evidences(max_dlas)

    toc = time.time()
    # very time consuming: ~ 4 mins for a single spectrum without parallelized.
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

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


def test_prior():
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

    log_priors = dla_gp.log_priors(z_qso, max_dlas=4)

    catalog_log_priors = np.array([-2.53774598, -4.97413739, -7.40285925, -9.74851888])

    assert np.all(np.abs(log_priors - catalog_log_priors) < 1e-4)


def prepare_dla_model(
    plate: int = 5309, mjd: int = 55929, fiber_id: int = 362
) -> DLAGPMAT:
    """
    Return a DLAGP instance from an input SDSS DR12 spectrum.
    """
    filename = "spec-{}-{}-{}.fits".format(plate, mjd, str(fiber_id).zfill(4))

    if not os.path.exists(filename):
        retrieve_raw_spec(plate, mjd, fiber_id)  # the spectrum at paper

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

    return dla_gp
