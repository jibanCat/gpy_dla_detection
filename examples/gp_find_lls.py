import os, re, sys
import numpy as np
from typing import Tuple
import time
import argparse
import itertools

# Use h5py to read the learned GP model
import h5py

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Compute the log of the sum of exponentials of input elements
# This function provides a stable way to compute log(exp(x) + exp(y))
from scipy.special import logsumexp

# plotting styles
import matplotlib as mpl
from matplotlib import pyplot as plt

# The test selection file includes a list of 100 quasar spectra and their redshifts
from tests import test_selection_fumagalli

# The module to read spectrum, which also helps us to download the file (for SDSS DR12 spectrum)
from gpy_dla_detection.read_spec import read_spec, retrieve_raw_spec, read_spec_dr14q

# The module to fit the spectrum
from gpy_dla_detection import voigt_lls as voigt

# The module to set the parameters
from gpy_dla_detection.lls_gp import LLSParameters

# The module to set the prior
from gpy_dla_detection.model_priors import PriorCatalog

# The module to set the GP model
from gpy_dla_detection.null_gp import NullGP

from gpy_dla_detection.lls_gp import LLSGPDR12
from gpy_dla_detection.lls_gp import LyaSamples, CIVSamples, MgIISamples

from gpy_dla_detection import voigt, voigt_mgii, voigt_civ

from scipy.integrate import (
    quad,
)  # to get the normalization constant for the probability density function

mpl.rcParams["figure.dpi"] = 150


# Null model GP : a GP model without any DLA intervening.
# Note that it's model without DLAs, so the noise term might still include some
# LLS or subDLAs
class NullGPDR12(NullGP):
    """
    Represents a Gaussian Process (GP) model for quasar spectra without Damped Lyman Alpha (DLA) systems
    using data from the SDSS DR12 survey. This class extends the NullGP class to incorporate
    learned parameters specific to the SDSS DR12 data set.

    Attributes:
        params (Parameters): Global parameters used for model training and defining priors.
        prior (PriorCatalog): Prior catalog based on the SDSS DR9 Lyman Alpha catalog.
        learned_file (str): Path to the .MAT file containing the learned GP model parameters.
        prev_tau_0 (float): Initial guess for the tau_0 parameter, affecting the mean flux of the GP mean function.
        prev_beta (float): Initial guess for the beta parameter, also affecting the mean flux.

    The class constructor loads the learned model parameters from a specified file and initializes
    the GP model with these parameters.
    """

    def __init__(
        self,
        params: LLSParameters,
        prior: PriorCatalog,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        prev_tau_0: float = 0.00554,
        prev_beta: float = 3.182,
    ):
        with h5py.File(learned_file, "r") as learned:
            rest_wavelengths = learned["rest_wavelengths"][:, 0]
            mu = learned["mu"][:, 0]
            M = learned["M"][()].T
            log_omega = learned["log_omega"][:, 0]
            log_c_0 = learned["log_c_0"][0, 0]
            log_tau_0 = learned["log_tau_0"][0, 0]
            log_beta = learned["log_beta"][0, 0]

        super().__init__(
            params,
            prior,
            rest_wavelengths,
            mu,
            M,
            log_omega,
            log_c_0,
            log_tau_0,
            log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
        )


def get_prior_catalog(param: LLSParameters):
    """
    Retrieves the prior catalog for the DLA analysis based on the provided parameters.

    Parameters:
        param (Parameters): Parameter instance containing settings for the analysis.

    Returns:
        PriorCatalog: An instance of the PriorCatalog class loaded with DLA data.

    This function is responsible for loading the DLA catalog which is used as a prior
    for the Gaussian Process modeling of quasar spectra.
    """
    # Note: you need to follow the README to download these files!
    prior = PriorCatalog(
        param,
        "data/dr12q/processed/catalog.mat",
        "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    )

    return prior


def get_default_parameters():
    """
    Generates a default set of parameters for the analysis and modeling of quasar spectra.

    Returns:
        Parameters: An instance of the Parameters class initialized with default settings.

    This function initializes parameters related to the spectrum loading, normalization,
    model range, and noise variance, among others, to their default values for the analysis.
    """

    # Note: these parameters are used during training. Change these parameters here won't change your
    # trained GP model unless you re-train your GP model.
    param = LLSParameters(
        loading_min_lambda=800,  # range of rest wavelengths to load  Å
        loading_max_lambda=1550,
        # normalization parameters
        normalization_min_lambda=1425,  # range of rest wavelengths to use   Å
        normalization_max_lambda=1475,  #   for flux normalization
        # null model parameters
        min_lambda=850.75,  # range of rest wavelengths to       Å
        max_lambda=1420.75,  #   model
        dlambda=0.25,  # separation of wavelength grid      Å
        k=20,  # rank of non-diagonal contribution
        max_noise_variance=3**2,  # maximum pixel noise allowed during model training
    )

    # BOSS DR12 effective optical depth
    param.tau_0_mu = 0.00554  # meanflux suppression for τ₀
    param.tau_0_sigma = 0.00064  # meanflux suppression for τ₀
    param.beta_mu = 3.182  # meanflux suppression for β
    param.beta_sigma = 0.074  # meanflux suppression for β

    return param


########################### Plotting Start #######################################
def plot_spectrum(wavelengths: np.ndarray, flux: np.ndarray, z_qso: float):
    """
    Plots the spectrum of a quasar given its wavelengths, flux, and redshift.

    Parameters:
        wavelengths (np.ndarray): The wavelengths of the quasar spectrum.
        flux (np.ndarray): The flux values of the quasar spectrum.
        z_qso (float): The redshift of the quasar.

    This function visualizes the quasar spectrum, highlighting the normalized flux across
    the rest-frame wavelengths and marking prominent emission lines for reference.
    """

    rest_wavelengths = wavelengths / (1 + z_qso)

    # Names and Rest-Wavelengths of the Emission Lines
    emission_rest_wavelengths = [1215.24, 912]
    emission_names = [r"Ly$\alpha$", r"Lyman Limit"]

    plt.figure(figsize=(16, 5))
    plt.plot(rest_wavelengths, flux / np.mean(flux))
    plt.xlabel("Rest-frame Wavelengths [$\AA$]")
    plt.ylabel("Normalized Flux")
    plt.ylim(-1, 5)
    plt.xlim(750, 1750)

    # plot the emission lines to help measure the z
    for n, w in zip(emission_names, emission_rest_wavelengths):
        plt.vlines(w, -1, 5, color="C3", ls="--")
        plt.text(w, 4, n, rotation="vertical", color="C3")


def plot_model(gp: NullGPDR12):
    """
    Plots the Gaussian Process model alongside the observed quasar spectrum.

    Parameters:
        gp (NullGPDR12): An instance of the NullGPDR12 class representing the GP model.

    This function visualizes the fit of the GP model to the observed quasar spectrum, including
    the mean function and uncertainty bounds, to assess the quality of the model.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    # Mean function
    ax.plot(
        gp.X,  # quasar spectrum's rest-frame wavelengths
        gp.Y,  # quasar spectrum's flux
        label="Data",
    )
    ax.fill_between(
        gp.X,
        gp.Y - 2 * np.sqrt(gp.v),
        gp.Y + 2 * np.sqrt(gp.v),
        label="SDSS Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )
    ax.plot(
        gp.rest_wavelengths,
        gp.mu,
        label="GP null model (mu = continuum)",
        color="C3",
        ls="--",
    )
    ax.plot(
        gp.X,
        gp.this_mu,
        label="GP null model (mu = meanflux)",  # GP model's mean function
    )
    ax.set_xlabel("Rest-frame Wavelengths [$\AA$]")
    ax.set_ylabel("Normalized Flux")
    ax.legend(loc="upper right")
    # Add the observed wavelengths to the top x-axis
    # with the correponding ticks of the bottom x-axis
    # top ticks = bottom ticks  * (1 + z_qso)
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(
        ["{:.0f}".format(tick * (1 + gp.z_qso)) for tick in ax.get_xticks()],
    )
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Observed-frame Wavelengths [$\AA$]")

    ax.set_ylim(-1, 5)
    fig.tight_layout()


def plot_lls_prior(prior: PriorCatalog):
    """
    Plots the prior probability of Lyman Limit Systems (LLS) as a function of quasar redshift (zQSO).

    Parameters:
        prior (PriorCatalog): An instance of PriorCatalog containing DLA/LLS prior information.
    """
    z_qso_list = np.linspace(2.15, 6)

    lls_prior = []
    for _z in z_qso_list:
        this_num_dlas, this_num_quasars = prior.less_ind(_z)

        # The prior of DLA = number of DLAs at a given redshift / (number of quasar sightlines with and without DLAs)
        _lls_prior = this_num_dlas / this_num_quasars
        lls_prior.append(_lls_prior)

    plt.plot(z_qso_list, lls_prior, label="p(LLS | zQSO)")
    plt.xlabel("zQSO")

    plt.ylabel("p(LLS | zQSO)")


def plot_sample_predictions(
    lya_gp: LLSGPDR12,
    gp: NullGPDR12,
    z_qso: float,
    log_posteriors: np.ndarray,
):
    """
    Plots the sample predictions for Lyman Alpha absorbers within a quasar spectrum.

    Parameters:
        nth_lya (int): The index of the absorber to plot.
        lya_gp (LLSGPDR12): An instance of LLSGPDR12 representing the GP model with LLS.
        gp (NullGPDR12): An instance of NullGPDR12 representing the GP model without LLS.
        z_qso (float): The redshift of the quasar.
    """
    # How many absorbers searches you want to plot
    # MAP absobers:
    i, j, k = np.unravel_index(np.nanargmax(log_posteriors), log_posteriors.shape)
    # nth_lya = max_Lya  # here we plot all of the searches

    sample_z_dlas = lya_gp.dla_samples.sample_z_dlas(
        lya_gp.this_wavelengths, lya_gp.z_qso
    )

    # Canvas with two panels
    fig, ax = plt.subplots(4, 1, figsize=(16, 20))

    # 1. Real spectrum space
    # Get the MAP values
    (
        MAP_log_nhi,
        MAP_z_lya,
        MAP_log_nmgii,
        MAP_z_mgiis,
        MAP_log_nciv,
        MAP_z_civ,
    ) = lya_gp.maximum_a_posteriori(log_posteriors)

    # feed in MAP values and get the absorption profile given (z_dlas, nhis)
    abs_lls = lya_gp.this_lls_gp(MAP_z_lya, 10**MAP_log_nhi)
    abs_mgii = lya_gp.this_mgii_gp(MAP_z_mgiis, 10**MAP_log_nmgii)
    abs_civ = lya_gp.this_civ_gp(MAP_z_civ, 10**MAP_log_nciv)

    # use the combined absorption profile to multiply to the GP model
    abs_combined = abs_lls * abs_mgii * abs_civ
    abs_mu = lya_gp.this_mu * abs_combined

    # Only plot the spectrum within the search range
    this_rest_wavelengths = lya_gp.x
    ind = this_rest_wavelengths < lya_gp.params.lya_wavelength
    this_rest_wavelengths = this_rest_wavelengths[ind]
    abs_mu = abs_mu[ind]

    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
        lya_gp.Y[ind],
    )
    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
        abs_mu,
        label=r"$\mathcal{M}$" + "LLS({i}) MgII({j}) CIV({k})".format(i=i, j=j, k=k),
        color="red",
    )
    ax[0].fill_between(
        (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
        gp.Y[ind] - 2 * np.sqrt(gp.v[ind]),
        gp.Y[ind] + 2 * np.sqrt(gp.v[ind]),
        label="SDSS Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )
    ax[0].plot(
        gp.rest_wavelengths * (1 + z_qso) / lya_gp.params.lya_wavelength - 1,
        gp.mu,
        label="GP null model (mu = continuum)",
        color="C3",
        ls="--",
    )

    # 2. Posterior space
    # 2.1. LLS
    # [color sequence] convert sample log likelihoods to values in (0, 1)
    sample_log_likelihoods = lya_gp.sample_log_likelihoods[
        :,
        1,
        0,
        0,
    ]  # only query the DLA(1) likelihoods
    # TODO: marginalize over k DLAs
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)
    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)
    # scale to make the colour more visible
    # TODO: make it more reasonable. scatter only takes values between [0, 1].
    colours = colours * 5 - 4
    colours[colours < 0] = 0
    # Scatter of posteriors
    ax[1].scatter(
        sample_z_dlas,
        lya_gp.dla_samples.log_nhi_samples,
        c=colours,
        marker=".",
        alpha=0.25,
    )
    # MAP estimate
    ax[1].scatter(
        MAP_z_lya,
        MAP_log_nhi,
        marker="*",
        s=100,
        color="C3",
    )
    # [min max sample zDLAs] instead of using min max from sample_z_dlas
    # using the zDLAs converted from wavelengths will better reflect the
    # range of wavelengths range in the this_mu plot.
    ax[1].set_xlim(
        (lya_gp.params.lyman_limit - 100) * (1 + z_qso) / lya_gp.params.lya_wavelength
        - 1,
        # sample_z_dlas.min(),
        z_qso,
    )
    ax[1].set_ylim(
        lya_gp.dla_samples.log_nhi_samples.min(),
        lya_gp.dla_samples.log_nhi_samples.max(),
    )
    ax[1].set_xlabel(r"$z_{Lya}$")
    ax[1].set_ylabel(r"$log N_{HI}$")

    # 2.2. MgII
    # [color sequence] convert sample log likelihoods to values in (0, 1)
    sample_log_likelihoods = lya_gp.sample_log_likelihoods[
        :,
        0,
        1,
        0,
    ]  # only query the DLA(1) likelihoods
    # TODO: marginalize over k DLAs
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)
    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)
    # scale to make the colour more visible
    # TODO: make it more reasonable. scatter only takes values between [0, 1].
    colours = colours * 5 - 4
    colours[colours < 0] = 0
    # Scatter of posteriors
    ax[2].scatter(
        lya_gp.sample_z_mgiis,
        lya_gp.mgii_samples.log_nmgii_samples,
        c=colours,
        marker=".",
        alpha=0.25,
    )
    # MAP estimate
    ax[2].scatter(
        MAP_z_mgiis,
        MAP_log_nmgii,
        marker="*",
        s=100,
        color="C3",
    )
    # [min max sample zDLAs] instead of using min max from sample_z_dlas
    # using the zDLAs converted from wavelengths will better reflect the
    # range of wavelengths range in the this_mu plot.
    ax[2].set_xlim(
        (lya_gp.params.lyman_limit - 100)
        * (1 + z_qso)
        / lya_gp.params.mgii_2803_wavelength
        - 1,
        # sample_z_dlas.min(),
        (lya_gp.params.lya_wavelength)
        * (1 + z_qso)
        / lya_gp.params.mgii_2803_wavelength
        - 1,
    )
    ax[2].set_ylim(
        lya_gp.mgii_samples.log_nmgii_samples.min(),
        lya_gp.mgii_samples.log_nmgii_samples.max(),
    )
    ax[2].set_xlabel(r"$z_{MgII}$")
    ax[2].set_ylabel(r"$log N_{MgII}$")

    # 2.3. CIV
    # [color sequence] convert sample log likelihoods to values in (0, 1)
    sample_log_likelihoods = lya_gp.sample_log_likelihoods[
        :,
        0,
        0,
        1,
    ]  # only query the DLA(1) likelihoods
    # TODO: marginalize over k DLAs
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)
    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)
    # scale to make the colour more visible
    # TODO: make it more reasonable. scatter only takes values between [0, 1].
    colours = colours * 5 - 4
    colours[colours < 0] = 0
    # Scatter of posteriors
    ax[3].scatter(
        lya_gp.sample_z_civs,
        lya_gp.civ_samples.log_nciv_samples,
        c=colours,
        marker=".",
        alpha=0.25,
    )
    # MAP estimate
    ax[3].scatter(
        MAP_z_civ,
        MAP_log_nciv,
        marker="*",
        s=100,
        color="C3",
    )
    # [min max sample zDLAs] instead of using min max from sample_z_dlas
    # using the zDLAs converted from wavelengths will better reflect the
    # range of wavelengths range in the this_mu plot.
    ax[3].set_xlim(
        (lya_gp.params.lyman_limit - 100)
        * (1 + z_qso)
        / lya_gp.params.civ_1550_wavelength
        - 1,
        # sample_z_dlas.min(),
        (lya_gp.params.lya_wavelength) * (1 + z_qso) / lya_gp.params.civ_1550_wavelength
        - 1,
    )
    ax[3].set_ylim(
        lya_gp.civ_samples.log_nciv_samples.min(),
        lya_gp.civ_samples.log_nciv_samples.max(),
    )
    ax[3].set_xlabel(r"$z_{CIV}$")
    ax[3].set_ylabel(r"$log N_{CIV}$")

    # You want the first panel has the same range
    ax[0].set_xlim(
        (lya_gp.params.lyman_limit - 100) * (1 + z_qso) / lya_gp.params.lya_wavelength
        - 1,
        # sample_z_dlas.min(),
        z_qso,
    )
    ax[0].legend()
    ax[0].set_ylim(-1, 5)

    return fig, ax


def plot_prediction_extended_spectrum(
    lya_gp: LLSGPDR12,
    gp: NullGPDR12,
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    log_posteriors: np.ndarray,
    z_lls_fumagalli: float,
):
    """
    Plots the extended spectrum prediction for a quasar, incorporating the effects of Lyman Alpha absorbers.

    Parameters:
        nth_lya (int): The index of the absorber to plot.
        lya_gp (LLSGPDR12): An instance of LLSGPDR12 representing the GP model with LLS.
        gp (NullGPDR12): An instance of NullGPDR12 representing the GP model without LLS.
        rest_wavelengths (np.ndarray): Array of rest wavelengths for the quasar spectrum.
        flux (np.ndarray): Array of flux values for the quasar spectrum.
        log_posteriors (np.ndarray): Array of log posteriors for the absorber searches.
        z_lls_fumagalli (float): The redshift of the Lyman Limit System from Fumagalli et al. (2011).
    """
    # MAP absobers:
    i, j, k = np.unravel_index(np.nanargmax(log_posteriors), log_posteriors.shape)

    # 1. Real spectrum space
    # Get the MAP values
    (
        MAP_log_nhi,
        MAP_z_lya,
        MAP_log_nmgii,
        MAP_z_mgiis,
        MAP_log_nciv,
        MAP_z_civ,
    ) = lya_gp.maximum_a_posteriori(log_posteriors)

    # Maximum posterior
    max_log_posterior = np.nanmax(log_posteriors)

    model_posteriors = np.exp(log_posteriors - max_log_posterior)
    model_posteriors = model_posteriors / np.nansum(model_posteriors)

    # feed in MAP values and get the absorption profile given (z_dlas, nhis)
    abs_lls = lya_gp.this_lls_gp(MAP_z_lya, 10**MAP_log_nhi)
    abs_mgii = lya_gp.this_mgii_gp(MAP_z_mgiis, 10**MAP_log_nmgii)
    abs_civ = lya_gp.this_civ_gp(MAP_z_civ, 10**MAP_log_nciv)

    # use the combined absorption profile to multiply to the GP model
    abs_combined = abs_lls * abs_mgii * abs_civ
    abs_mu = lya_gp.this_mu * abs_combined

    # Only plot the spectrum within the search range
    this_rest_wavelengths = lya_gp.x
    ind = this_rest_wavelengths < lya_gp.params.lya_wavelength
    this_rest_wavelengths = this_rest_wavelengths[ind]
    abs_mu = abs_mu[ind]

    fig, ax = plt.subplots(1, 1, figsize=(25, 8))

    # Spectrum space
    ax.plot(
        rest_wavelengths,
        flux / gp.normalization_median,
    )
    ax.plot(
        gp.rest_wavelengths,
        gp.mu,
        label="GP null model (mu = continuum)",
        color="C3",
        ls="--",
    )

    # Model with the MAP values
    ax.plot(
        this_rest_wavelengths,
        abs_mu,
        label=r"$\mathcal{M}$"
        + "LLS({i}) MgII({j}) CIV({k})".format(i=i, j=j, k=k)
        + " Maximum Posterior {:.2g}".format(model_posteriors[i, j, k]),
        color="red",
        # lw=2,
    )
    ax.fill_between(
        this_rest_wavelengths,
        gp.Y[ind] - 2 * np.sqrt(gp.v[ind]),
        gp.Y[ind] + 2 * np.sqrt(gp.v[ind]),
        label="SDSS Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )
    # Adding MAP lines to the plot
    ax = add_MAP_vlines(
        lya_gp,
        MAP_z_lya,
        MAP_log_nhi,
        MAP_z_mgiis,
        MAP_log_nmgii,
        MAP_z_civ,
        MAP_log_nciv,
        ax,
        lya_gp.z_qso,
    )
    ax = add_Fumagalli_vlines(ax, z_lls_fumagalli, lya_gp.z_qso)

    ax.set_xlim(750, 1550)
    ax.set_ylim(-2, 5)
    plt.legend()
    plt.xlabel("Rest-frame Wavelengths [$\AA$]", fontdict={"fontsize": 16})
    plt.ylabel("Normalized Flux", fontdict={"fontsize": 16})
    plt.tight_layout()

    # Add the observed wavelengths to the top x-axis
    # with the correponding ticks of the bottom x-axis
    # top ticks = bottom ticks  * (1 + z_qso)
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(
        ["{:.0f}".format(tick * (1 + lya_gp.z_qso)) for tick in ax.get_xticks()],
    )
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Observed-frame Wavelengths [$\AA$]", fontdict={"fontsize": 16})
    return fig, ax


def add_MAP_vlines(
    lya_gp: LLSGPDR12,
    MAP_z_lya: np.ndarray,
    MAP_log_nhi: np.ndarray,
    MAP_z_mgiis: np.ndarray,
    MAP_log_nmgii: np.ndarray,
    MAP_z_civ: np.ndarray,
    MAP_log_nciv: np.ndarray,
    ax: plt.Axes,
    z_qso: float,
):
    """
    Adds vertical lines to the plot for the maximum a posteriori (MAP) redshifts of absorbers.

    Parameters:
        MAP_z_lya (float): The MAP redshift of the Lyman Alpha absor
        MAP_log_nhi (float): The MAP log column density of the Lyman Alpha absorser.
        MAP_z_mgiis (float): The MAP redshift of the MgII absorber.
        MAP_log_nmgii (float): The MAP log column density of the MgII absorber.
        MAP_z_civ (float): The MAP redshift of the CIV absorber.
        MAP_log_nciv (float): The MAP log column density of the CIV absorber.
        ax (Axes): The matplotlib Axes object to add the lines to.
    """
    # Add Vertical line at to the redshifts of LLS
    for z, n in zip(MAP_z_lya, MAP_log_nhi):
        # For all ly series lines
        ax.vlines(
            [
                (1 + z) * lya_gp.params.lya_wavelength / (1 + z_qso),
                (1 + z) * voigt.transition_wavelengths[1] * 1e8 / (1 + z_qso),
                (1 + z) * voigt.transition_wavelengths[2] * 1e8 / (1 + z_qso),
                # (1 + z) * voigt.transition_wavelengths[3] * 1e8 / (1 + z_qso),
                (1 + z) * lya_gp.params.lyman_limit / (1 + z_qso),
            ],
            -2,
            5,
            color="C1",
            ls="--",
            lw=2,
        )
        # Add text to the redshifts of LLS
        labels = [
            r"$\leftarrow Ly\alpha$",
            r"$\leftarrow Ly\beta$",
            r"$\leftarrow Ly\gamma$",
            r"$\leftarrow Ly\infty$",
        ]
        for i in range(3):
            ax.text(
                (1 + z) * voigt.transition_wavelengths[i] * 1e8 / (1 + z_qso),
                2.5,
                labels[i] + ": z={:.3g}".format(z) + " ln={:.3g}".format(n),
                rotation=90,
                color="C1",
                fontdict={"fontsize": 16},
            )

        ax.text(
            (1 + z) * lya_gp.params.lyman_limit / (1 + z_qso),
            2.0,
            labels[-1] + ": z={:.3g}".format(z) + " ln={:.3g}".format(n),
            rotation=90,
            color="C3",
            fontdict={"fontsize": 16},
        )

        # Add lines for the Metalicity of the LLS
        metal_lines = [1397.61, 1549.48, 1908.734, 2799.117]
        ax.vlines(
            [
                (1 + z) * metal_lines[0] / (1 + z_qso),
                (1 + z) * metal_lines[1] / (1 + z_qso),
                # (1 + z) * metal_lines[2] / (1 + z_qso),
                # (1 + z) * metal_lines[3] / (1 + z_qso),
            ],
            -2,
            5,
            color="C4",
            ls="dashdot",
            lw=2,
        )
        labels = [
            r"$\leftarrow SiIV$",
            r"$\leftarrow CIV$",
            # r"$\leftarrow CIII$",
            # r"$\leftarrow MgII$",
        ]
        for i in range(2):
            ax.text(
                (1 + z) * metal_lines[i] / (1 + z_qso),
                2.5,
                labels[i] + ": {:.3g}".format(z),
                rotation=90,
                color="C4",
                fontdict={"fontsize": 16},
            )

    # Add Vertical line at to the redshifts of MgII
    for z, n in zip(MAP_z_mgiis, MAP_log_nmgii):
        ax.vlines(
            [
                (1 + z) * voigt_mgii.transition_wavelengths[0] * 1e8 / (1 + z_qso),
                (1 + z) * voigt_mgii.transition_wavelengths[1] * 1e8 / (1 + z_qso),
            ],
            -2,
            5,
            color="C2",
            ls="--",
            lw=2,
        )
        # Add text to the redshifts of MgII
        for i in range(1):
            ax.text(
                (1 + z) * voigt_mgii.transition_wavelengths[i] * 1e8 / (1 + z_qso),
                2.5,
                "$\leftarrow$" + " MgII: z={:.3g}".format(z) + " ln={:.3g}".format(n),
                rotation=90,
                color="C2",
                fontdict={"fontsize": 16},
            )
    # Add Vertical line at to the redshifts of CIV
    for z, n in zip(MAP_z_civ, MAP_log_nciv):
        ax.vlines(
            [
                (1 + z) * voigt_civ.transition_wavelengths[0] * 1e8 / (1 + z_qso),
                (1 + z) * voigt_civ.transition_wavelengths[1] * 1e8 / (1 + z_qso),
            ],
            -2,
            5,
            color="C4",
            ls="--",
            lw=2,
        )
        # Add text to the redshifts of CIV
        for i in range(1):
            ax.text(
                (1 + z) * voigt_civ.transition_wavelengths[i] * 1e8 / (1 + z_qso),
                2.5,
                "$\leftarrow$" + "CIV: z={:.3g}".format(z) + " lnN={:.3g}".format(n),
                rotation=90,
                color="C4",
                fontdict={"fontsize": 16},
            )

    return ax


def add_Fumagalli_vlines(ax: plt.Axes, z_lls: float, z_qso: float):
    """
    Adds vertical lines to the plot for the redshifts of absorbers based on Fumagalli et al. (2011).

    Parameters:
        ax (Axes): The matplotlib Axes object to add the lines to.
        z_lls (float): The redshift of the Lyman Limit System (LLS) absorber.
    """
    # Fumagalli et al. (2020) absorber redshifts
    # the redshift of Lyman limit systems
    ax.vlines(
        [
            (1 + z_lls) * LLSParameters().lya_wavelength / (1 + z_qso),
            (1 + z_lls) * LLSParameters().lyman_limit / (1 + z_qso),
        ],
        -2,
        5,
        color="C4",
        ls="dotted",
        lw=2,
    )
    # Add text to the redshifts of LLS
    labels = [
        r"$\leftarrow$ (F) $Ly\alpha$",
        r"$\leftarrow$ (F) $Ly\infty$",
    ]
    ax.text(
        (1 + z_lls) * LLSParameters().lya_wavelength / (1 + z_qso),
        -1.9,
        labels[0] + ": z={:.3g}".format(z_lls),
        rotation=270,
        color="C5",
        fontdict={"fontsize": 16},
    )

    ax.text(
        (1 + z_lls) * LLSParameters().lyman_limit / (1 + z_qso),
        -1.9,
        labels[-1] + ": z={:.3g}".format(z_lls),
        rotation=270,
        color="C5",
        fontdict={"fontsize": 16},
    )

    return ax


########################### Plotting End #######################################


def save_processed_file(
    filename: str,
    gp: NullGPDR12,
    lya_gp: LLSGPDR12,
    log_likelihoods: np.ndarray,
    z_qso: float,
    max_lls: int,
    max_mgii: int,
    max_civ: int,
):
    """
    Saves the processed data from the GP models and Lya absorber analysis into an HDF5 file.

    Parameters:
        filename (str): The path to the file where the data will be saved.
        gp (NullGPDR12): An instance of NullGPDR12 representing the GP model without LLS.
        lya_gp (LLSGPDR12): An instance of LLSGPDR12 representing the GP model with LLS.
        log_likelihoods (np.ndarray): Log likelihoods of the absorber models.
        z_qso (float): The redshift of the quasar.
        max_lls (int): The maximum number of Lyman Limit Systems (LLS) to consider.
        max_mgii (int): The maximum number of MgII absorbers to consider.
        max_civ (int): The maximum number of CIV absorbers to consider.
    """
    # Calculate the priors for absorptions: Lya, MgII, CIV
    # Shape: (max + 1), where the first element is 0
    log_priors_lls = np.append(np.array([0]), lya_gp.log_priors(z_qso, max_lls))
    log_priors_mgii = np.append(np.array([0]), lya_gp.log_priors_mgii(z_qso, max_mgii))
    log_priors_civ = np.append(np.array([0]), lya_gp.log_priors_civ(z_qso, max_civ))
    # Calculate the null model prior
    log_prior_null = gp.log_prior(z_qso)
    log_priors = np.empty((max_lls + 1, max_mgii + 1, max_civ + 1))
    log_priors[:] = np.nan
    log_priors[0, 0, 0] = log_prior_null
    combinations = [
        (i, j, k)
        for i, j, k in itertools.product(
            range(0, max_lls + 1), range(0, max_mgii + 1), range(0, max_civ + 1)
        )
    ]
    for comb in combinations:
        i, j, k = comb
        log_priors[i, j, k] = log_priors_lls[i] + log_priors_mgii[j] + log_priors_civ[k]

    # Calculate posteriors for the null model and Lya models.
    log_posteriors = log_likelihoods + log_priors

    max_log_posterior = np.nanmax(log_posteriors)

    model_posteriors = np.exp(log_posteriors - max_log_posterior)
    model_posteriors = model_posteriors / np.nansum(model_posteriors)

    (
        MAP_log_nhi,
        MAP_z_lya,
        MAP_log_nmgii,
        MAP_z_mgiis,
        MAP_log_nciv,
        MAP_z_civ,
    ) = lya_gp.maximum_a_posteriori(log_posteriors)
    # Get the indices of the maximum log posterior
    i, j, k = np.unravel_index(np.nanargmax(log_posteriors), log_posteriors.shape)

    # write into HDF5 file
    with h5py.File(filename, "w") as f:
        # storing default parameter settings for the model
        f.create_dataset(
            "prior_z_qso_increase", data=lya_gp.params.prior_z_qso_increase
        )
        f.create_dataset("k", data=lya_gp.params.k)
        f.create_dataset(
            "normalization_min_lambda", data=lya_gp.params.normalization_min_lambda
        )
        f.create_dataset(
            "normalization_max_lambda", data=lya_gp.params.normalization_max_lambda
        )
        f.create_dataset("min_z_cut", data=lya_gp.params.min_z_cut)
        f.create_dataset("max_z_cut", data=lya_gp.params.max_z_cut)
        f.create_dataset("num_lya_samples", data=lya_gp.params.num_dla_samples)
        f.create_dataset("num_lines", data=lya_gp.params.num_lines)
        f.create_dataset("num_forest_lines", data=lya_gp.params.num_forest_lines)

        # storing the sampling variables

        # Save storage, not saving the full inference, only the MAP
        f.create_dataset(
            "sample_log_likelihoods", data=lya_gp.sample_log_likelihoods[:, i, j, k]
        )
        # f.create_dataset("base_sample_inds", data=lya_gp.base_sample_inds)

        f.create_dataset("log_priors", data=log_priors)
        f.create_dataset("log_likelihoods", data=log_likelihoods)
        f.create_dataset("log_posteriors", data=log_posteriors)

        f.create_dataset("MAP_z_lya", data=MAP_z_lya)
        f.create_dataset("MAP_log_nhi", data=MAP_log_nhi)
        f.create_dataset("MAP_z_mgii", data=MAP_z_mgiis)
        f.create_dataset("MAP_log_nmgii", data=MAP_log_nmgii)
        f.create_dataset("MAP_z_civ", data=MAP_z_civ)
        f.create_dataset("MAP_log_nciv", data=MAP_log_nciv)

        f.create_dataset("model_posteriors", data=model_posteriors)

        # also save zQSOs
        f.create_dataset("z_qsos", data=np.array([z_qso]))

    return (i, j, k), model_posteriors, log_posteriors


########################### Main Sampling Function ######################################
def main(
    nspec: int,
    max_lls: int = 2,
    max_mgii: int = 2,
    max_civ: int = 2,
    num_lines: int = 4,
    img_dir: str = "images-lls-metals/",
    lls_sample_h5: str = "../data/dr12q/processed/hi_samples.h5",
    mgii_sample_h5: str = "../data/dr12q/processed/mgii_samples.h5",
    civ_sample_h5: str = "../data/dr12q/processed/civ_samples.h5",
):
    """
    Main function to execute the analysis pipeline for detecting LLS and DLA in quasar spectra.

    Parameters:
        nspec (int): Index of the spectrum to analyze.
        max_lls (int): Maximum number of Lyman Limit Systems (LLS) to consider in the analysis.
        max_mgii (int): Maximum number of MgII absorbers to consider in the analysis.
        max_civ (int): Maximum number of CIV absorbers to consider in the analysis.
        num_lines (int): Number of Lyman series lines to use in the Voigt profile.
        img_dir (str): Directory to save generated images.
        lls_sample_h5 (str): Path to the file containing LLS samples.

    This function includes the loading of quasar spectra, the application of Gaussian Process
    models, and the detection and analysis of LLS and DLA features within the spectra. Results are
    visualized and saved to the specified directory.
    """

    loader = test_selection_fumagalli.QuasarTableLoader()
    loader.load_data()

    # take a spectrum and download it
    filename = "{}.fits".format(loader.quasar_name[nspec])
    z_qso = loader.redshift[nspec]

    # make sure the directory exists
    os.makedirs(img_dir, exist_ok=True)
    img_dir = os.path.join(img_dir, "nspec_{}".format(nspec))
    os.makedirs(img_dir, exist_ok=True)

    # Save the Fumagalli quasar spectrum information into readable text file
    print(
        "[Info] Saving the Fumagalli quasar spectrum information into readable text file ..."
    )
    # print out the Fagamaili quasar spectrum information
    print("Quasar Name: {}".format(loader.quasar_name[nspec]))
    print("Redshift: {}".format(z_qso))
    print("SN_1150A: {}".format(loader.SN_1150A[nspec]))
    print("Classification Outcome: {}".format(loader.classification_outcome[nspec]))
    print("Note:  (1: quasar with LLS; 2: quasar without LLS; 4: non quasar)")
    with open(os.path.join(img_dir, "Fumagalli_quasar_info.txt"), "w") as f:
        f.write("Quasar Name: {}\n".format(loader.quasar_name[nspec]))
        f.write("Redshift: {}\n".format(z_qso))
        f.write("RA: {}\n".format(loader.right_ascension_deg[nspec]))
        f.write("DEC: {}\n".format(loader.declination_deg[nspec]))
        f.write("SN_1150A: {}\n".format(loader.SN_1150A[nspec]))
        f.write("Science Primary: {}\n".format(loader.science_primary[nspec]))
        f.write("In Training Set: {}\n".format(loader.in_training_set[nspec]))
        f.write(
            "Classification Outcome: {}\n".format(loader.classification_outcome[nspec])
        )
        f.write("Note:  (1: quasar with LLS; 2: quasar without LLS; 4: non quasar)\n")
        f.write("LLS Redshift: {}\n".format(loader.LLS_redshift[nspec]))

    # If we haven't downloaded the file, this cell will help you download the file from SDSS database
    base_rawspectra_dir = os.path.join("data", "raw_spectra")
    filename = os.path.join(base_rawspectra_dir, filename)

    if not os.path.exists(filename):
        # This line gets the plate, mjd, and fiber_id from the given filename
        # Note: re is the regex.
        plate, mjd, fiber_id = re.findall(
            r"spec-([0-9]+)-([0-9]+)-([0-9]+).fits",
            filename,
        )[0]
        # Download the file using the given plate, mjd, and fiber_id
        # If the file doesn't exist, try to download the file from dr12q
        try:
            # Download the file using the given plate, mjd, and fiber_id
            retrieve_raw_spec(
                int(plate),
                int(mjd),
                int(fiber_id),
                release="dr14q",
                base_dir=base_rawspectra_dir,
            )
        except Exception as e:
            print("Error: ", e)
            print("Downloading from dr14q ...")
            retrieve_raw_spec(
                int(plate),
                int(mjd),
                int(fiber_id),
                release="dr12q",
                base_dir=base_rawspectra_dir,
            )

    # make sure the file exists
    assert os.path.exists(filename) == True

    # read the spectrum
    # wavelengths is x
    # flux is y
    # A spectrum is flux(wavelength)
    print("[Info] Reading the quasar spectrum ...")
    wavelengths, flux, noise_variance, pixel_mask = read_spec_dr14q(filename)

    rest_wavelengths = wavelengths / (1 + z_qso)

    # Preview the spectrum
    plot_spectrum(wavelengths, flux, z_qso)
    plt.savefig(os.path.join(img_dir, "spectrum.png"), dpi=150, format="png")
    plt.clf()
    plt.close()

    # Get the default parameters and the prior
    print("[Info] Getting the default parameters and the prior ...")
    param = get_default_parameters()
    prior = get_prior_catalog(param)

    # Plot the LLS prior
    plot_lls_prior(prior)
    plt.savefig(os.path.join(img_dir, "lls_prior.png"), dpi=150, format="png")
    plt.clf()
    plt.close()

    # get the instance of GP null model
    print("[Info] Getting the instance of GP null model ...")
    gp = NullGPDR12(
        params=param,
        prior=prior,
        # you should put your downloaded file in this directory
        learned_file="data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        # SDSS/BOSS DR12 meanflux effective optical depth for Lyman alpha forest
        prev_tau_0=0.00554,  # suppression: tau
        prev_beta=3.182,  # suppression: beta
    )

    # Make the GP model interpolated onto the observed quasar spectum
    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # Plot the GP model
    plot_model(gp)
    plt.savefig(os.path.join(img_dir, "model.png"), dpi=150, format="png")
    plt.clf()
    plt.close()

    # Get the LLS samples
    # samples for the Monte Carlo integration to get the likelihood of an absorber
    # p( D | absorber model ) = \int p( D | z_abs, NHI, absorber model) p(z_abs) p(NHI) d z_abs d NHI
    print("[Info] Reading the LLS samples ...")
    # HI absorption samples
    with h5py.File(lls_sample_h5, "r") as f:
        halton_sequence = f["halton_sequence"][()]
        samples_log_nhis = f["samples_log_nhis"][()]
    # MgII absorption samples
    with h5py.File(mgii_sample_h5, "r") as f:
        halton_sequence_mgii = f["halton_sequence"][()]
        samples_log_mgiis = f["samples_log_mgiis"][()]
    # CIV absorption samples
    with h5py.File(civ_sample_h5, "r") as f:
        halton_sequence_civ = f["halton_sequence"][()]
        samples_log_civs = f["samples_log_civs"][()]

    # LyaSamples: this is the prior of absorber redshifts and column densities
    # p(z_abs) p(NHI)
    lya_samples = LyaSamples(
        params=param,
        prior=prior,
        # offset samples
        offset_samples=halton_sequence,
        log_nhi_samples=samples_log_nhis,
    )
    plt.hist(samples_log_nhis, bins=100, density=True, label="Monte Carlo samples")
    plt.xlabel("log NHI")
    plt.ylabel("Probability density function")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "monte_carlo_samples.png"), dpi=150, format="png")
    plt.clf()
    plt.close()
    # MgII samples
    mgii_samples = MgIISamples(
        params=param,
        prior=prior,
        offset_samples=halton_sequence_mgii,
        log_mgii_samples=samples_log_mgiis,
    )
    plt.hist(samples_log_mgiis, bins=100, density=True, label="Monte Carlo samples")
    plt.xlabel("log N_MgII")
    plt.ylabel("Probability density function")
    plt.legend()
    plt.savefig(
        os.path.join(img_dir, "monte_carlo_samples_mgii.png"), dpi=150, format="png"
    )
    plt.clf()
    plt.close()
    # CIV samples
    civ_samples = CIVSamples(
        params=param,
        prior=prior,
        offset_samples=halton_sequence_civ,
        log_civ_samples=samples_log_civs,
    )
    plt.hist(samples_log_civs, bins=100, density=True, label="Monte Carlo samples")
    plt.xlabel("log N_CIV")
    plt.ylabel("Probability density function")
    plt.legend()
    plt.savefig(
        os.path.join(img_dir, "monte_carlo_samples_civ.png"), dpi=150, format="png"
    )
    plt.clf()
    plt.close()

    # Lya Model GP : this is null model GP + Voigt profile, which is parameterised with
    # {(z_lya, logNHI)}_{i=1}^{k} parameters. k stands for maximum Lya absorpbers we want to search.
    # we will compute log posteriors for DLA(1), ..., DLA(k) models.
    # I want the Voigt profile to use Lya series lines up to 10
    param.num_lines = num_lines
    param.num_dla_samples = len(samples_log_nhis)
    print("[Info] Getting the Lya Model GP ...")
    lya_gp = LLSGPDR12(
        params=param,
        prior=prior,
        lya_samples=lya_samples,  # 1. you input the NHI zabs samples to integrate
        mgii_samples=mgii_samples,  # 2. you input the NMgII zabs samples to integrate
        civ_samples=civ_samples,  # 3. you input the NCIV zabs samples to integrate
        min_z_separation=2000,  # 4. you have the minimum zabs separateion, in unit of km/s
        learned_file="data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        broadening=True,
        prev_tau_0=0.00554,  # suppression: tau
        prev_beta=3.182,  # suppression: beta
    )
    lya_gp.set_data(
        rest_wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso,
        build_model=True,
    )

    # search up to max_Lya absorbers
    print("[Info] Do the sampling ...")
    tt = time.time()
    # This is the line does the integral
    log_likelihoods_lya = lya_gp.log_model_evidences(
        max_lls=max_lls, max_mgiis=max_mgii, max_civs=max_civ
    )
    # Shape: (max_LLS, max_MgII, max_CIV)

    print("Spend {:.4g} seconds".format(time.time() - tt))

    # Save the processed data into an HDF5 file
    print("[Info] Saving the processed data into an HDF5 file ...")
    (i, j, k), model_posteriors, log_posteriors = save_processed_file(
        os.path.join(img_dir, "processed.h5"),
        gp,
        lya_gp,
        log_likelihoods_lya,
        z_qso,
        max_lls,
        max_mgii,
        max_civ,
    )

    # MAP estimates
    (
        MAP_log_nhi,
        MAP_z_lya,
        MAP_log_nmgii,
        MAP_z_mgiis,
        MAP_log_nciv,
        MAP_z_civ,
    ) = lya_gp.maximum_a_posteriori(log_posteriors)

    # Plot the sample predictions
    print("[Info] Plotting the sample predictions ...")
    plot_sample_predictions(
        lya_gp,
        gp,
        z_qso,
        log_posteriors,
    )
    plt.savefig(os.path.join(img_dir, "fsample_predictions.png"), dpi=150, format="png")
    plt.clf()
    plt.close()

    # Plot the extended spectrum prediction
    plot_prediction_extended_spectrum(
        lya_gp,
        gp,
        rest_wavelengths,
        flux,
        log_posteriors,
        z_lls_fumagalli=loader.LLS_redshift[nspec],
    )
    plt.savefig(
        os.path.join(img_dir, "extended_predictions.png"), dpi=150, format="png"
    )
    plt.clf()
    plt.close()

    # Save the basic information of the LLs detection run:
    with open(os.path.join(img_dir, "LLS_detection_info.txt"), "w") as f:
        f.write("Spectrum: {}\n".format(filename))
        f.write("Max LLS, MgII, CIV: {}, {}, {}\n".format(max_lls, max_mgii, max_civ))
        f.write("Number of Lya Series Lines: {}\n".format(num_lines))
        f.write("LLS Sample H5: {}\n".format(lls_sample_h5))
        f.write("MgII Sample H5: {}\n".format(mgii_sample_h5))
        f.write("CIV Sample H5: {}\n".format(civ_sample_h5))
        f.write("Processed File: {}\n".format(os.path.join(img_dir, "processed.h5")))
        # Number of LLS samples used
        f.write("Number of Samples: {}\n".format(len(samples_log_nhis)))

        # Detected number of absorbers and model posteriors, and the redshifts and nhis
        f.write(
            "Detected Number of Absorbers: LLS {}, MgII {}, CIV {}\n".format(i, j, k)
        )
        f.write("Model Posteriors: \n{}\n".format(model_posteriors))
        f.write("Max Model Posteriors: {}\n".format(np.nanmax(model_posteriors)))
        f.write("MAP z_lyas: {}\n".format(MAP_z_lya))
        f.write("MAP log_nhis: {}\n".format(MAP_log_nhi))
        f.write("MAP z_mgiis: {}\n".format(MAP_z_mgiis))
        f.write("MAP log_nmgii: {}\n".format(MAP_log_nmgii))
        f.write("MAP z_civs: {}\n".format(MAP_z_civ))
        f.write("MAP log_nciv: {}\n".format(MAP_log_nciv))

        # Time taken for the detection
        f.write("Time Taken: {:.4g} seconds\n".format(time.time() - tt))

        # The directory where the images are saved
        f.write("Image Directory: {}\n".format(img_dir))
        # The directory where the processed file is saved
        f.write("Processed File: {}\n".format(os.path.join(img_dir, "processed.h5")))


if __name__ == "__main__":
    # Set the parser for the script and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nspec",
        type=int,
        help="The index of the spectrum to be analyzed.",
    )
    parser.add_argument(
        "--max_lls",
        type=int,
        default=4,
        help="The maximum number of LLS absorbers to search.",
    )
    parser.add_argument(
        "--max_mgii",
        type=int,
        default=4,
        help="The maximum number of MgII absorbers to search.",
    )
    parser.add_argument(
        "--max_civ",
        type=int,
        default=4,
        help="The maximum number of CIV absorbers to search.",
    )
    parser.add_argument(
        "--num_lines",
        type=int,
        default=4,
        help="The number of Lya series lines to use in the Voigt profile.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="images-lls/",
        help="The directory to save the images.",
    )
    parser.add_argument(
        "--lls_sample_h5",
        type=str,
        default="data/dr12q/processed/lls_samples.h5",
        help="The file containing the LLS samples.",
    )
    parser.add_argument(
        "--mgii_sample_h5",
        type=str,
        default="data/dr12q/processed/mgii_samples.h5",
        help="The file containing the MgII samples.",
    )
    parser.add_argument(
        "--civ_sample_h5",
        type=str,
        default="data/dr12q/processed/civ_samples.h5",
        help="The file containing the CIV samples.",
    )
    args = parser.parse_args()

    main(
        args.nspec,
        args.max_lls,
        args.max_mgii,
        args.max_civ,
        args.num_lines,
        args.img_dir,
        args.lls_sample_h5,
        args.mgii_sample_h5,
        args.civ_sample_h5,
    )
    # example: python examples/gp_find_lls.py --nspec 0 --max_lls 2 --max_mgii 2 --max_civ 2 --num_lines 4 --img_dir images-lls/ --lls_sample_h5 data/dr12q/processed/lls_samples.h5 --mgii_sample_h5 data/dr12q/processed/mgii_samples.h5 --civ_sample_h5 data/dr12q/processed/civ_samples.h5
