"""
Generate logN samples from column density distribution function (CDDF).

CDDF : f(NHI, z) = d^2N/dz/dNHI

Usually parameterized as a power-law function of the form:
f(NHI, z) = A(z) * (NHI/N0)^beta

where A(z) is the normalization factor, N0 is the pivot column density, and beta is the power-law index.
"""

import numpy as np

# We want Monte Carlo samples - here to use Halton sequence to speed up the convergence
from scipy.stats.qmc import Halton
from scipy.interpolate import interp1d
from scipy.integrate import (
    quad,
)  # to get the normalization constant for the probability density function


# Define the CDDF parameters
# DLA CDDF: Prochaska & Wolfe 2009
# broken power law
# for log(N_HI) < 21.75: beta = -1.8
# for log(N_HI) > 21.75: beta = -3
log_nd = 21.75
log_k_dla = np.log10(7) - 25
log_cddf_dla = lambda log_n, beta_dla: log_k_dla + beta_dla * (log_n - log_nd)

# Super Lyman Limit Systems (SLLS) CDDF
# for 19 < log(N_HI) < 20.3: beta = -1.2
log_k_slls = log_cddf_dla(20.3, -1.8) + 1.2 * 20.3
log_cddf_slls = lambda log_n: log_k_slls - 1.2 * log_n

# Lyman Limit Systems (LLS) CDDF
# for 17.5 < log(N_HI) < 19: beta = -0.8
log_k_lls = log_cddf_slls(19) + 0.8 * 19
log_cddf_lls = lambda log_n: log_k_lls - 0.8 * log_n

# MgII CDDF: Churchill et al. 2020
# for 12 < log(N_MgII) < 15: beta = -1.45
# doppler parameter b = 5.7 km/s
log_k_mgii = (
    -12 + 1.45 * 14
)  # TODO: This is approximate value, need to send email ask Churchill
log_cddf_mgii = lambda log_n: log_k_mgii - 1.45 * log_n

# CIV CDDF: Kim et al. 2003
# for 12 < log(N_CIV) < 15: beta = -1.85
log_k_civ = 11.41
log_cddf_civ = lambda log_n: log_k_civ - 1.85 * log_n


def dla_normalized_pdf(log_n):
    """
    Compute the normalized probability density function (PDF) for the DLA CDDF.

    Parameters:
    - log_n: The column density in log10(N_HI) space.

    Returns:
    - pdf: The normalized PDF at the given column density.
    """
    # directly CDDF for DLA, SLLS, LLS
    unnormalized_pdf = lambda log_nhi: (
        10 ** log_cddf_lls(log_nhi) * (log_nhi < 19) * (log_nhi >= 17.5)
        + 10 ** log_cddf_slls(log_nhi) * (log_nhi >= 19) * (log_nhi < 20.3)
        + 10 ** log_cddf_dla(log_nhi, -1.8) * (log_nhi < 21.75) * (log_nhi >= 20.3)
        + 10 ** log_cddf_dla(log_nhi, -3) * (log_nhi >= 21.75)
    )

    Z = quad(unnormalized_pdf, 17.5, 23.0)[0]

    return unnormalized_pdf(log_n) / Z


def mgii_normalized_pdf(log_n):
    """
    Compute the normalized probability density function (PDF) for the MgII CDDF.

    Parameters:
    - log_n: The column density in log10(N_MgII) space.

    Returns:
    - pdf: The normalized PDF at the given column density.
    """
    # CDDF for MgII
    unnormalized_pdf = lambda log_nmgii: 10 ** log_cddf_mgii(log_nmgii)

    Z = quad(unnormalized_pdf, 14, 16)[0]

    return unnormalized_pdf(log_n) / Z


def civ_normalized_pdf(log_n):
    """
    Compute the normalized probability density function (PDF) for the CIV CDDF.

    Parameters:
    - log_n: The column density in log10(N_CIV) space.

    Returns:
    - pdf: The normalized PDF at the given column density.
    """
    # CDDF for CIV
    unnormalized_pdf = lambda log_nciv: 10 ** log_cddf_civ(log_nciv)

    Z = quad(unnormalized_pdf, 14, 16)[0]

    return unnormalized_pdf(log_n) / Z


def inverse_transform_sampling(
    pdf: callable,
    domain: tuple,
    halton_sequence: np.ndarray,
    resolution: int = 10000,
):
    """
    Generates samples from a given probability density function (pdf) using the inverse transform sampling method.

    Parameters:
    - pdf: A callable representing the probability density function.
    - domain: A tuple (min, max) specifying the range over which the pdf is defined.
    - halton_sequence: A NumPy array of Halton sequence points.
    - resolution: The number of points to use for approximating the inverse CDF.

    Returns:
    - samples: A NumPy array of samples generated from the pdf.
    """

    # Validate inputs
    if domain[0] >= domain[1]:
        raise ValueError(
            "Invalid domain. Ensure that the domain min is less than the domain max."
        )

    # Generate points within the domain and compute the PDF
    x = np.linspace(domain[0], domain[1], num=resolution)
    y = pdf(x)

    if np.any(y < 0):
        raise ValueError("PDF values must be non-negative.")

    # Compute the cumulative distribution function (CDF) and normalize
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y / cdf_y[-1]  # Normalize to 1

    # Create the inverse CDF through interpolation
    inverse_cdf = interp1d(
        cdf_y, x, bounds_error=False, fill_value=(domain[0], domain[1])
    )

    # Generate uniform samples in the CDF range and convert to samples from the PDF
    samples = inverse_cdf(halton_sequence)

    return samples


def generate_dla_samples(num_samples: int, resolution: int = 10000):
    """
    Generate samples from the DLA CDDF using inverse transform sampling.

    Parameters:
    - num_samples: The number of samples to generate.

    Returns:
    - samples: A NumPy array of samples from the DLA CDDF.
    """
    # Define the domain of the DLA CDDF
    domain = (17.5, 23.0)
    sampler = Halton(
        d=2,  # dimension of parameter space
        scramble=False,
    )
    halton_sequence = sampler.random(num_samples)
    # Generate samples using inverse transform sampling
    samples = inverse_transform_sampling(
        dla_normalized_pdf,
        domain,
        halton_sequence[:, 0],
        resolution,
    )

    return samples, halton_sequence[:, 1]


def generate_mgii_samples(num_samples: int, resolution: int = 10000):
    """
    Generate samples from the MgII CDDF using inverse transform sampling.

    Parameters:
    - num_samples: The number of samples to generate.

    Returns:
    - samples: A NumPy array of samples from the MgII CDDF.
    """
    # Define the domain of the MgII CDDF
    domain = (14.0, 16.0)
    sampler = Halton(
        d=2,  # dimension of parameter space
        scramble=False,
    )
    halton_sequence = sampler.random(num_samples)
    # Generate samples using inverse transform sampling
    samples = inverse_transform_sampling(
        mgii_normalized_pdf,
        domain,
        halton_sequence[:, 0],
        resolution,
    )

    return samples, halton_sequence[:, 1]


def generate_civ_samples(num_samples: int, resolution: int = 10000):
    """
    Generate samples from the CIV CDDF using inverse transform sampling.

    Parameters:
    - num_samples: The number of samples to generate.

    Returns:
    - samples: A NumPy array of samples from the CIV CDDF.
    """
    # Define the domain of the CIV CDDF
    domain = (14.0, 16.0)
    sampler = Halton(
        d=2,  # dimension of parameter space
        scramble=False,
    )
    halton_sequence = sampler.random(num_samples)
    # Generate samples using inverse transform sampling
    samples = inverse_transform_sampling(
        civ_normalized_pdf,
        domain,
        halton_sequence[:, 0],
        resolution,
    )

    return samples, halton_sequence[:, 1]
