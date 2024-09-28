from numba import njit
import numpy as np
from scipy.special import wofz

# physical constants
c = 2.99792458e10  # speed of light in cm/s

# Instrumental profile (can be passed as an argument, or defined globally)
instrument_profile = np.array(
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


@njit
def voigt_jit(x, sigma, gamma):
    """
    JIT-compiled Voigt profile calculation using Numba.
    """
    z = (x + 1j * gamma) / (np.sqrt(2) * sigma)
    return np.real(wofz(z)) / (np.sqrt(2 * np.pi) * sigma)


@njit
def voigt_absorption_jit(
    wavelengths,
    nhi,
    z_dla,
    num_lines,
    sigma,
    gammas,
    leading_constants,
    transition_wavelengths,
    broadening=True,
):
    """
    JIT-compiled version of the Voigt absorption profile.

    Parameters:
    ----------
    wavelengths : np.ndarray
        Observed wavelengths (Å).
    nhi : float
        Column density of this absorber (cm⁻²).
    z_dla : float
        Redshift of this absorber.
    num_lines : int
        Number of Lyman series members.
    sigma : float
        Gaussian broadening parameter.
    gammas : np.ndarray
        Lorentzian width parameters.
    leading_constants : np.ndarray
        Precomputed constants for each transition line.
    transition_wavelengths : np.ndarray
        Wavelengths of Lyman transitions.
    broadening : bool
        Whether to apply instrumental broadening.

    Returns:
    --------
    profile : np.ndarray
        The Voigt absorption profile.
    """
    num_points = wavelengths.shape[0]
    profile = np.zeros(num_points - len(instrument_profile) + 1)

    # Build the multipliers for the relative velocity
    multipliers = c / (transition_wavelengths[:num_lines] * (1 + z_dla)) / 1e8

    # Precompute raw profiles
    total = np.empty((num_lines, num_points))

    for l in range(num_lines):
        velocity = wavelengths * multipliers[l] - c
        total[l, :] = -leading_constants[l] * voigt_jit(velocity, sigma, gammas[l])

    # Compute raw profile (taking the sum over lines)
    raw_profile = np.exp(nhi * np.nansum(total, axis=0))

    # Apply broadening if necessary
    if broadening:
        profile[:] = np.convolve(raw_profile, instrument_profile, "valid")
        return profile
    else:
        return raw_profile


# Define the Voigt absorption wrapper function
def compute_voigt_absorption(wavelengths, nhi, z_dla, num_lines=3, broadening=True):
    """
    Compute the Voigt absorption profile for a DLA system.

    This function acts as a wrapper to invoke the JIT-compiled function.
    """

    # Lorentzian widths:
    #   gammas[i] = Gammas[i] * transition_wavelengths[i] / (4 * M_PI);
    gammas: np.ndarray = np.array(
        [
            6.06075804241938613e02,  # cm s⁻¹
            1.54841462408931704e02,
            6.28964942715328164e01,
            3.17730561586147395e01,
            1.82838676775503330e01,
            9.15463131005758157e00,
            6.08448802613156925e00,
            4.24977523573725779e00,
            3.08542121666345803e00,
            2.31184525202557767e00,
            1.77687796208123139e00,
            1.39477990932179852e00,
            1.11505539984541979e00,
            9.05885451682623022e-01,
            7.45877170715450677e-01,
            6.21261624902197052e-01,
            5.22994533400935269e-01,
            4.44469874827484512e-01,
            3.80923210837841919e-01,
            3.28912390446060132e-01,
            2.85949711597237033e-01,
            2.50280032040928802e-01,
            2.20224061101442048e-01,
            1.94686521675913549e-01,
            1.73082093051965591e-01,
            1.54536566013816490e-01,
            1.38539175663870029e-01,
            1.24652675945279762e-01,
            1.12585442799479921e-01,
            1.02045988802423507e-01,
            9.27433783998286437e-02,
        ]
    )

    # quantities of Lyman series
    transition_wavelengths: np.ndarray = np.array(
        [  # lambda_ul, cm
            1.2156701e-05,  # Lya
            1.0257223e-05,  # Lyb ...
            9.725368e-06,
            9.497431e-06,
            9.378035e-06,
            9.307483e-06,
            9.262257e-06,
            9.231504e-06,
            9.209631e-06,
            9.193514e-06,
            9.181294e-06,
            9.171806e-06,
            9.16429e-06,
            9.15824e-06,
            9.15329e-06,
            9.14919e-06,
            9.14576e-06,
            9.14286e-06,
            9.14039e-06,
            9.13826e-06,
            9.13641e-06,
            9.13480e-06,
            9.13339e-06,
            9.13215e-06,
            9.13104e-06,
            9.13006e-06,
            9.12918e-06,
            9.12839e-06,
            9.12768e-06,
            9.12703e-06,
            9.12645e-06,
        ]
    )
    # leading constants[i] =
    #        M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c)
    leading_constants: np.ndarray = np.array(
        [  # cm²
            1.34347262962625339e-07,
            2.15386482180851912e-08,
            7.48525170087141461e-09,
            3.51375347286007472e-09,
            1.94112336271172934e-09,
            1.18916112899713152e-09,
            7.82448627128742997e-10,
            5.42930932279390593e-10,
            3.92301197282493829e-10,
            2.92796010451409027e-10,
            2.24422239410389782e-10,
            1.75895684469038289e-10,
            1.40338556137474778e-10,
            1.13995374637743197e-10,
            9.37706429662300083e-11,
            7.79453203101192392e-11,
            6.55369055970184901e-11,
            5.58100321584169051e-11,
            4.77895916635794548e-11,
            4.12301389852588843e-11,
            3.58872072638707592e-11,
            3.12745536798214080e-11,
            2.76337116167110415e-11,
            2.44791750078032772e-11,
            2.15681362798480253e-11,
            1.93850080479346101e-11,
            1.72025364178111889e-11,
            1.55051698336865945e-11,
            1.40504672409331934e-11,
            1.28383057589411395e-11,
            1.16264059622218997e-11,
        ]
    )

    # Garnett (2016): the width of Gaussian is fixed, with
    # the assumption that the gas temperature fixed to 10^4 K
    # this imparts a thermal broadening of 13 km s⁻¹
    sigma: float = 9.08537121627923800e05  # cm s⁻¹

    # Call the JIT-compiled function
    return voigt_absorption_jit(
        wavelengths,
        nhi,
        z_dla,
        num_lines,
        sigma,
        gammas,
        leading_constants,
        transition_wavelengths,
        broadening,
    )
