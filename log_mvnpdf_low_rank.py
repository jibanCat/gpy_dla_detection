import numpy as np
import scipy
import scipy.linalg
from profilehooks import profile

from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.read_spec import read_spec

@profile
def log_mvnpdf_low_rank(
    y: np.ndarray, mu: np.ndarray, M: np.ndarray, d: np.ndarray
) -> float:
    """
    efficiently computes
    
        log N(y; mu, MM' + diag(d))
    
    :param y: this_flux, (n_points, )
    :param mu: this_mu, the mean vector of GP, (n_points, )
    :param M: this_M, the low rank decomposition of covariance matrix, (n_points, k)
    :param d: diagonal noise term, (n_points, )
    """
    log_2pi = 1.83787706640934534

    n, k = M.shape

    y = y[:, None] - mu[:, None]

    d_inv = 1 / d[:, None]  # (n_points, 1)
    D_inv_y = d_inv * y  # (n_points, 1)
    D_inv_M = d_inv * M  # (n_points, k)

    # use Woodbury identity, define
    #   B = (I + M' D^-1 M),
    # then
    #   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1
    B = np.matmul(M.T, D_inv_M)  # (k, n_points) * (n_points, k) -> (k, k)
    # add the identity matrix with magic indicing
    B.ravel()[0 :: (k + 1)] = B.ravel()[0 :: (k + 1)] + 1
    # numpy cholesky returns lower triangle, different than MATLAB's upper triangle
    L = np.linalg.cholesky(B)
    # C = B^-1 M' D^-1
    tmp = scipy.linalg.solve_triangular(L, D_inv_M.T, lower=True)  # (k, n_points)
    C = scipy.linalg.solve_triangular(L.T, tmp, lower=False)  # (k, n_points)

    K_inv_y = D_inv_y - np.matmul(D_inv_M, np.matmul(C, y))  # (n_points, 1)

    log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))

    log_p = -0.5 * (np.matmul(y.T, K_inv_y).sum() + log_det_K + n * log_2pi)

    return log_p

if __name__ == "__main__":
    # y = np.array([1, 2])
    # mu = np.array([1, 2])
    # M = np.array([[2, 3, 1], [1, 2, 4]])
    # d = np.eye(2) * 2

    # test 1
    filename = "spec-5309-55929-0362.fits"

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

    print(log_mvnpdf_low_rank(dla_gp.y, dla_gp.this_mu, dla_gp.M, dla_gp.v + dla_gp.this_omega2))
