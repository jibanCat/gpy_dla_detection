"""
effective_optical_depth.py : calculate the total effective optical
from Hydrogen lines
"""
import numpy as np

from . import voigt


def effective_optical_depth(
    wavelengths: np.ndarray,
    beta: float,
    tau_0: float,
    z_qso: float,
    num_forest_lines: int,
    skip_lya_indicator: bool = True,
) -> np.ndarray:
    """
    calculate the total optical depth:
    
    effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
    
    where 
    
    1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)

    :param wavelengths: λobs, observed wavelengths.
    :param beta: β, exponent of the effective optical depth.
    :param tau_0: τ, scale factor of the effective optical depth.
    :param z_qso: quasar redshift.
    :param num_forest_lines: number of forest lines want to implement.
    :param skip_lya_indicator: for DLA detection case, the modelling range is slightly
        larger than lya_wavelength, so if we apply the indicator in the lya optical depth,
        then the lya peak region will not be covered and the result will look artificial.
        This should set to False if we want to model to include the metal line region.
    
    :return total_optical_depth: exp(-total_optical_depth) will be the mean-flux suppression.
    """
    # change unit from cm to A
    transition_wavelengths = voigt.transition_wavelengths * 1e8
    oscillator_strengths = voigt.oscillator_strengths

    lya_wavelength = transition_wavelengths[0]
    lya_oscillator_strength = oscillator_strengths[0]

    # To count the effect of Lyman series from higher z,
    # we compute the absorbers' redshifts for all members of the series
    this_lyseries_zs = np.empty((wavelengths.shape[0], num_forest_lines))
    this_lyseries_zs[:] = np.nan

    for i in range(num_forest_lines):
        this_lyseries_zs[:, i] = (
            wavelengths - transition_wavelengths[i]
        ) / transition_wavelengths[i]

    # Lyman series absorption effect on the mean-flux
    # apply the lya_absorption after the interpolation because NaN will appear in this_mu
    total_optical_depth = np.empty((wavelengths.shape[0], num_forest_lines))
    total_optical_depth[:] = np.nan

    for i in range(num_forest_lines):
        # calculate the oscillator strength for this lyman series member
        this_tau_0 = (
            tau_0
            * oscillator_strengths[i]
            / lya_oscillator_strength
            * transition_wavelengths[i]
            / lya_wavelength
        )

        total_optical_depth[:, i] = this_tau_0 * (1 + this_lyseries_zs[:, i]) ** beta

        # indicator function: z absorbers <= z_qso
        # here is different from multi-dla processing script
        # I choose to use zero instead or nan to indicate
        # values outside of the Lyman forest
        indicator = this_lyseries_zs[:, i] <= z_qso
        total_optical_depth[:, i] = total_optical_depth[:, i] * indicator

    return total_optical_depth
