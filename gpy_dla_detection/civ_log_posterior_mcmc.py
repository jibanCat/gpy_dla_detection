"""
A set of simple functions to evaluate the log posteriors of
given parameter priors
"""
from typing import Tuple

import numpy as np
import scipy
from scipy.linalg import lapack

from .voigt_civ import voigt_absorption


def log_prior(
    z_civ: float,
    log_nciv: float,
    sigma: float,
    min_z_civ: float,
    max_z_civ: float,
    min_log_nciv: float,
    max_log_nciv: float,
    min_sigma: float,
    max_sigma: float,
    pdf,
) -> float:
    """
    log prior probability for a single CIV intervening
    with the quasar emission model.

    Current only uniform prior for zCIV.
    TODO: Add Reza's data driven prior
    """
    cond = (
        lambda z_civ, log_nciv, sigma:
          (z_civ < max_z_civ)
        * (z_civ > min_z_civ)
        * (log_nciv > min_log_nciv)
        * (log_nciv < max_log_nciv)
        * (sigma > min_sigma)
        * (sigma < max_sigma)
    )

    if cond(z_civ, log_nciv, sigma):
        return np.log(pdf(log_nciv))

    return -np.inf


def log_posterior(
    theta: Tuple,
    this_wavelengths: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    z_qso: float,
    min_z_civ: float,
    max_z_civ: float,
    min_log_nciv: float,
    max_log_nciv: float,
    min_sigma : float,
    max_sigma : float,
    pdf,
    padded_wavelengths: np.ndarray,
    this_mu: np.ndarray,
    this_M: np.ndarray,
    pixel_mask: np.ndarray,
    ind_unmasked: np.ndarray,
    num_lines: np.ndarray,
) -> float:
    """
    get the posterior probability on a given pair of samples.

    :param theta: tuple, (z_civ, log_nciv, sigma).
    :param civ_gp: the CIV model, input as an argument.

    :return log_posterior:
    """
    z_civ, log_nciv, sigma = theta

    lp = log_prior(z_civ, log_nciv, sigma,
        min_z_civ=min_z_civ, max_z_civ=max_z_civ,
        min_log_nciv=min_log_nciv, max_log_nciv=max_log_nciv,
        min_sigma=min_sigma, max_sigma=max_sigma,
        pdf=pdf,
    )

    if not np.isfinite(lp):
        return -np.inf

    log_like = sample_log_likelihood_k_civs(
        np.array([z_civ]),
        10**np.array([log_nciv]),
        np.array([sigma]),
        y=y,
        v=v,
        padded_wavelengths=padded_wavelengths,
        this_mu=this_mu,
        this_M=this_M,
        pixel_mask=pixel_mask,
        ind_unmasked=ind_unmasked,
        num_lines=num_lines,
    )
    return lp + log_like


# just need to get the likelihood function linearly independent
def sample_log_likelihood_k_civs(
    z_civ: np.ndarray,
    nciv: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    padded_wavelengths: np.ndarray,
    this_mu: np.ndarray,
    this_M: np.ndarray,
    pixel_mask: np.ndarray,
    ind_unmasked: np.ndarray,
    num_lines: np.ndarray,
) -> float:
    """
    Compute the log likelihood of CIV within a quasar spectrum:
        p(y | λ, σ², M, ω, {z_civ, logNCIV})

    :param z_civ: an array of z_civ you want to condition on
    :param nciv: an array of nciv you want to condition on
    """
    assert len(z_civ) == len(nciv)

    civ_mu, civ_M = this_civ_gp(
        z_civ=z_civ,
        nciv=nciv,
        sigma=sigma,
        padded_wavelengths=padded_wavelengths,
        this_mu=this_mu,
        this_M=this_M,
        pixel_mask=pixel_mask,
        ind_unmasked=ind_unmasked,
        num_lines=num_lines,
    )

    sample_log_likelihood = log_mvnpdf_low_rank(y, civ_mu, civ_M, v)

    return sample_log_likelihood


def this_civ_gp(
    z_civ: np.ndarray,
    nciv: np.ndarray,
    sigma: np.ndarray,
    padded_wavelengths: np.ndarray,
    this_mu: np.ndarray,
    this_M: np.ndarray,
    pixel_mask: np.ndarray,
    ind_unmasked: np.ndarray,
    num_lines: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the CIV GP model with k intervening civ profiles onto
    the mean and covariance.

    :param z_civ: (k_civs, ), the redshifts of intervening CIVs
    :param nciv: (k_civs, ), the column densities of intervening CIVs
    :param sigma: (k_civs, )

    :return: (civ_mu, civ_M)
    :return civ_mu: (n_points, ), the GP mean model
    :return civ_M: (n_points, k), the GP covariance

    Note: the number of Voigt profile lines is controlled by self.params : Parameters,
    I prefer to not to allow users to change from the function arguments since that
    would easily cause inconsistent within a pipeline. But if a user want to change
    the num_lines, they can change via changing the instance attr of the self.params:Parameters
    like:
        self.params.num_lines = <the number of lines preferred to be used>
    This would happen when a user want to know whether the result would converge with increasing
    number of lines.
    """
    assert len(z_civ) == len(nciv)

    k_civs = len(z_civ)

    # to retain only unmasked pixels from computed absorption profile
    mask_ind = ~pixel_mask[ind_unmasked]

    # absorption corresponding to this sample
    absorption = voigt_absorption(
        padded_wavelengths, z_civ=z_civ[0], nciv=nciv[0], sigma=sigma[0], num_lines=num_lines,
    )

    # absorption corresponding to other CIVs in multiple CIV samples
    for j in range(1, k_civs):
        absorption = absorption * voigt_absorption(
            padded_wavelengths, z_civ=z_civ[j], nciv=nciv[j], sigma=sigma[j], num_lines=num_lines,
        )

    absorption = absorption[mask_ind]

    assert len(absorption) == len(this_mu)

    civ_mu = this_mu * absorption
    civ_M = this_M * absorption[:, None]

    return civ_mu, civ_M


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
