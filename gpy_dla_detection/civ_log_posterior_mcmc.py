"""
A set of simple functions to evaluate the log posteriors of
given parameter priors
"""
from typing import Tuple

import numpy as np
import scipy
from scipy.linalg import lapack

from .dla_samples import DLASamplesMAT
from .voigt import voigt_absorption

# from .dla_gp import DLAGP


def log_prior(
    z_dla: float,
    log_nhi: float,
    min_z_dla: float,
    max_z_dla: float,
    min_log_nhi: float,
    max_log_nhi: float,
    pdf,
) -> float:
    """
    log prior probability for a single DLA intervening
    with the quasar emission model.

    Uniform prior for zDLA and a data-driven prior for logNHI. 
    
    p(theta|zQSO) = p(zDLA|zQSO) p(NHI)    
    """
    cond = (
        lambda z_dla, log_nhi: (z_dla < max_z_dla)
        * (z_dla > min_z_dla)
        * (log_nhi > min_log_nhi)
        * (log_nhi < max_log_nhi)
    )

    if cond(z_dla, log_nhi):
        return np.log(pdf(log_nhi))
    return -np.inf


def log_posterior(
    theta: Tuple,
    this_wavelengths: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    z_qso: float,
    min_z_dla: float,
    max_z_dla: float,
    min_log_nhi: float,
    max_log_nhi: float,
    pdf,
    padded_wavelengths: np.ndarray,
    this_mu: np.ndarray,
    this_M: np.ndarray,
    this_omega2: np.ndarray,
    pixel_mask: np.ndarray,
    ind_unmasked: np.ndarray,
    num_lines: np.ndarray,
) -> float:
    """
    get the posterior probability on a given pair of samples.

    TODO: make this function to be multi-DLA.

    :param theta: tuple, (z_dla, log_nhi).
    :param dla_gp: the DLA model, input as an argument. Maybe slow down the computation
        but it is prettier.
    
    :return log_posterior:
    """
    z_dla, log_nhi = theta

    lp = log_prior(z_dla, log_nhi, min_z_dla, max_z_dla, min_log_nhi, max_log_nhi, pdf,)

    if not np.isfinite(lp):
        return -np.inf

    log_like = sample_log_likelihood_k_dlas(
        np.array([z_dla]),
        10**np.array([log_nhi]),
        y=y,
        v=v,
        padded_wavelengths=padded_wavelengths,
        this_mu=this_mu,
        this_M=this_M,
        this_omega2=this_omega2,
        pixel_mask=pixel_mask,
        ind_unmasked=ind_unmasked,
        num_lines=num_lines,
    )
    return lp + log_like


# just need to get the likelihood function linearly independent
def sample_log_likelihood_k_dlas(
    z_dlas: np.ndarray,
    nhis: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    padded_wavelengths: np.ndarray,
    this_mu: np.ndarray,
    this_M: np.ndarray,
    this_omega2: np.ndarray,
    pixel_mask: np.ndarray,
    ind_unmasked: np.ndarray,
    num_lines: np.ndarray,
) -> float:
    """
    Compute the log likelihood of k DLAs within a quasar spectrum:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, {z_dla, logNHI}_{i=1}^k)

    :param z_dlas: an array of z_dlas you want to condition on
    :param nhis: an array of nhis you want to condition on
    """
    assert len(z_dlas) == len(nhis)

    dla_mu, dla_M, dla_omega2 = this_dla_gp(
        z_dlas=z_dlas,
        nhis=nhis,
        padded_wavelengths=padded_wavelengths,
        this_mu=this_mu,
        this_M=this_M,
        this_omega2=this_omega2,
        pixel_mask=pixel_mask,
        ind_unmasked=ind_unmasked,
        num_lines=num_lines,
    )

    sample_log_likelihood = log_mvnpdf_low_rank(y, dla_mu, dla_M, dla_omega2 + v)

    return sample_log_likelihood


def this_dla_gp(
    z_dlas: np.ndarray,
    nhis: np.ndarray,
    padded_wavelengths: np.ndarray,
    this_mu: np.ndarray,
    this_M: np.ndarray,
    this_omega2: np.ndarray,
    pixel_mask: np.ndarray,
    ind_unmasked: np.ndarray,
    num_lines: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the DLA GP model with k intervening DLA profiles onto
    the mean and covariance.

    :param z_dlas: (k_dlas, ), the redshifts of intervening DLAs
    :param nhis: (k_dlas, ), the column densities of intervening DLAs

    :return: (dla_mu, dla_M, dla_omega2)
    :return dla_mu: (n_points, ), the GP mean model with k_dlas DLAs intervening.
    :return dla_M: (n_points, k), the GP covariance with k_dlas DLAs intervening.
    :return dla_omega2: (n_points), the absorption noise with k_dlas DLAs intervening.S

    Note: the number of Voigt profile lines is controlled by self.params : Parameters,
    I prefer to not to allow users to change from the function arguments since that
    would easily cause inconsistent within a pipeline. But if a user want to change
    the num_lines, they can change via changing the instance attr of the self.params:Parameters
    like:
        self.params.num_lines = <the number of lines preferred to be used>
    This would happen when a user want to know whether the result would converge with increasing
    number of lines.
    """
    assert len(z_dlas) == len(nhis)

    k_dlas = len(z_dlas)

    # to retain only unmasked pixels from computed absorption profile
    mask_ind = ~pixel_mask[ind_unmasked]

    # absorption corresponding to this sample
    absorption = voigt_absorption(
        padded_wavelengths, z_dla=z_dlas[0], nhi=nhis[0], num_lines=num_lines,
    )

    # absorption corresponding to other DLAs in multiple DLA samples
    for j in range(1, k_dlas):
        absorption = absorption * voigt_absorption(
            padded_wavelengths, z_dla=z_dlas[j], nhi=nhis[j], num_lines=num_lines,
        )

    absorption = absorption[mask_ind]

    assert len(absorption) == len(this_mu)

    dla_mu = this_mu * absorption
    dla_M = this_M * absorption[:, None]
    dla_omega2 = this_omega2 * absorption ** 2

    return dla_mu, dla_M, dla_omega2


def log_mvnpdf_low_rank(
    y: np.ndarray,
    mu: np.ndarray,
    M: np.ndarray,
    d: np.ndarray,
    scipy_lapack: bool = True,
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
    if scipy_lapack:
        tmp = np.matmul(lapack.dtrtri(np.asfortranarray(L), lower=1)[0], D_inv_M.T)
        C = np.matmul(lapack.dtrtri(np.asfortranarray(L.T), lower=0)[0], tmp)
    else:
        tmp = scipy.linalg.solve_triangular(L, D_inv_M.T, lower=True)  # (k, n_points)
        C = scipy.linalg.solve_triangular(L.T, tmp, lower=False)  # (k, n_points)

    K_inv_y = D_inv_y - np.matmul(D_inv_M, np.matmul(C, y))  # (n_points, 1)

    log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))

    log_p = -0.5 * (np.matmul(y.T, K_inv_y).sum() + log_det_K + n * log_2pi)

    return log_p
