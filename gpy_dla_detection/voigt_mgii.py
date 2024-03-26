"""
voigt.py : python version of the Voigt profile for 
    Reza's voigt.c file, including:

- A doublet MgII absorption line
- parameterized using : (1) z_mgii, (2) nmgii;
- input wavelengths: observed wavelengths

Note:
Want to check how's the instrumental project affecting the absorption dips
"""
import numpy as np
from scipy.special import wofz

# physical constants in cgs
c: float = 2.99792458e10  # speed of light          cm s⁻¹
k: float = 1.38064852e-16  # Boltzmann constant      erg K⁻¹
m_p: float = 1.672621898e-24  # proton mass             g
m_e: float = 9.10938356e-28  # electron mass           g
e: float = 4.803204672997660e-10  # elementary charge       statC

# MgII doublet
transition_wavelengths: np.ndarray = np.array(
    [  # lambda_ul, cm
        2796.3542699e-8,
        2803.5314853e-8,
    ]
)

oscillator_strengths: np.ndarray = np.array(
    [  # oscillator strengths f_ul, dimensionless
        0.6155,
        0.3058,
    ]
)

Gammas: np.ndarray = np.array(
    [  # transition rates s⁻¹
        2.68e8,
        2.66e8,
    ]
)

# leading constants[i] =
#        M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c)
leading_constants: np.ndarray = np.array(
    [  # cm²
        4.567963400781244e-07,
        2.2753346785102732e-07,
    ]
)

# Lorentzian widths:
#   gammas[i] = Gammas[i] * transition_wavelengths[i] / (4 * M_PI);
gammas: np.ndarray = np.array(
    [
        596.3718302855555,
        593.4405390190135,
    ]
)

# fixed width of convolution
width: int = 3  # dimensionless

# instrumental profile
instrument_profile: np.ndarray = np.array(
    [
        2.17460992138080811e-03,
        4.11623059580451742e-02,
        2.40309364651846963e-01,
        4.32707438937454059e-01,  # center pixel
        2.40309364651846963e-01,
        4.11623059580451742e-02,
        2.17460992138080811e-03,
    ]
)


def Gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    G(x; sigma) = 1 / sqrt(2 pi sigma^2) * exp( - x^2 / 2 sigma^2 )
    """
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x**2) / 2 / sigma**2)


def Lorentzian(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    L(x; gamma) = (gamma / pi ) / (x**2 + gamma**2)
    """
    return gamma / np.pi / (x**2 + gamma**2)


def Voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """
    Vogit line profile

    V(x; sigma, gamma) = Re[ w(z) ] / sqrt( 2 pi sigma^2 )
    """
    z = (x + 1j * gamma) / (np.sqrt(2) * sigma)
    return np.real(wofz(z)) / (np.sqrt(2 * np.pi) * sigma)


def voigt_absorption(
    wavelengths: np.ndarray,
    nmgii: float,
    z_mgii: float,
    sigma: float,
    num_lines: int = 2,
    broadening: bool = True,
) -> np.ndarray:
    """
    Voigt line profile for absorptions

    Parameters:
    ----
    wavelengths (np.ndarray) : observed wavelengths (Å)
    nmgii (float) : column density of MgII absorber (cm⁻2)
    z_mgii (float) : redshift of MgII absorber

    raw_profile =
        exp( nhi * ( - leading_constants[j] * Voigt(velocity, sigma, gammas[j] ) )  )

    for the relative velocity:
    velocity =
        c * ( wavelengths * / ( transition_wavelengths[j] * (1 + z) ) - 1 )

    for the leading constants:
    leading_constants[i] =
       M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c)


    /* instrumental broadening */
    for (i = 0; i < num_points; i++)
        for (j = i, k = 0; j <= i + 2 * width; j++, k++)
        profile[i] += raw_profile[j] * instrument_profile[k];

    Note:
    ----
    unit conversion from cm to A is 10^-8
    """
    # number of pixels within the input spectrum
    num_points = wavelengths.shape[0]

    # initialize a profile
    # absorption profile : dimensionless
    profile = np.zeros((num_points - 2 * width))

    # raw_profile before convolve with the instrumental profile
    raw_profile = np.empty((num_points,))

    # build the multipliers for the relative velocity
    multipliers = c / (transition_wavelengths[:num_lines] * (1 + z_mgii)) / 1e8

    # compute raw Voigt profile
    total = np.empty((num_lines, raw_profile.shape[0]))

    for l in range(num_lines):
        velocity = wavelengths * multipliers[l] - c

        total[l, :] = -leading_constants[l] * Voigt(velocity, sigma, gammas[l])

    raw_profile[:] = np.exp(np.float(nmgii) * np.nansum(total, axis=0))

    if broadening:
        # num_points = len(profile)

        # # instrumental broadening
        # for i in range(num_points):
        #     for k,j in enumerate(range(i, i + 2 * width + 1)):
        #         profile[i] += raw_profile[j] * instrument_profile[k]
        # return  profile
        profile[:] = np.convolve(raw_profile, instrument_profile, "valid")
        return profile

    return raw_profile
