import os, re, sys
import numpy as np
from typing import Tuple
import time
import argparse

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
from gpy_dla_detection.set_parameters import Parameters

# The module to set the prior
from gpy_dla_detection.model_priors import PriorCatalog

# The module to set the GP model
from gpy_dla_detection.null_gp import NullGP

from gpy_dla_detection.dla_gp import DLAGP
from gpy_dla_detection.dla_samples import DLASamples, DLASamplesMAT

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
        params: Parameters,
        prior: PriorCatalog,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
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


# Strong Lyman Alpha model GP : a GP model with at least one strong Lyman alpha absorption.
class LLSGPDR12(DLAGP):
    """
    Represents a Gaussian Process (GP) model for quasar spectra with at least one strong Lyman Alpha
    absorption, utilizing data from the SDSS DR12 survey. This class extends the DLAGP class to
    specifically model and analyze the presence of Lyman Limit Systems (LLS) within the spectra.

    Attributes:
        lya_samples (DLASamplesMAT): Monte Carlo samples of prior distributions for NHI and zLya.
        params (Parameters): Global parameters used for model training and defining priors.
        prior (PriorCatalog): Prior catalog based on the SDSS DR9 Lyman Alpha catalog.
        learned_file (str): Path to the .MAT file containing the learned GP model parameters.
        prev_tau_0 (float): Initial guess for the tau_0 parameter, affecting the mean flux.
        prev_beta (float): Initial guess for the beta parameter, also affecting the mean flux.
        min_z_separation (float): Minimum redshift separation in km/s to consider between LLS.
        broadening (bool): Indicates if instrumental broadening is considered in the model.

    The class constructor loads the learned model parameters from a specified file and initializes
    the GP model, incorporating the Voigt profile parameterization for strong Lyman alpha absorbers.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        lya_samples: DLASamplesMAT,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
        min_z_separation: float = 3000.0,  # km/s
        broadening: bool = True,
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
            lya_samples,
            rest_wavelengths,
            mu,
            M,
            log_omega,
            log_c_0,
            log_tau_0,
            log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )

    def this_dla_gp(
        self, z_lls: np.ndarray, nhis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the LLS GP model with k intervening LLS profiles onto
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
        assert len(z_lls) == len(nhis)

        k_lls = len(z_lls)

        # to retain only unmasked pixels from computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # [broadening] use the padded wavelengths for convolution
        # otherwise, should use unmasked wavelengths.
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        # absorption corresponding to this sample
        absorption = voigt.voigt_absorption(
            wavelengths,
            z_lls=z_lls[0],
            nhi=nhis[0],
            num_lines=self.params.num_lines,
            broadening=self.broadening,  # the switch for instrumental broadening controlled by instance attr
        )

        # absorption corresponding to other DLAs in multiple DLA samples
        for j in range(1, k_lls):
            absorption = absorption * voigt.voigt_absorption(
                wavelengths,
                z_lls=z_lls[j],
                nhi=nhis[j],
                num_lines=self.params.num_lines,
                broadening=self.broadening,  # the switch for instrumental broadening controlled by instance attr
            )

        absorption = absorption[mask_ind]

        assert len(absorption) == len(self.this_mu)

        lls_mu = self.this_mu * absorption
        lls_M = self.this_M * absorption[:, None]
        lls_omega2 = self.this_omega2 * absorption**2

        return lls_mu, lls_M, lls_omega2


class LyaSamples(DLASamples):
    """
    Represents parameter priors for strong Lyman Alpha absorbers within quasar spectra. This class
    provides functionality to sample from these priors for the purposes of modeling and detecting
    Lyman Alpha systems.

    Attributes:
        params (Parameters): Parameter instance containing settings for the analysis.
        prior (PriorCatalog): Prior catalog instance for obtaining DLA information.
        offset_samples (np.ndarray): Samples for offsets used in Monte Carlo integration.
        log_nhi_samples (np.ndarray): Logarithm of column density samples for Monte Carlo integration.
        max_z_cut (float): Maximum redshift cut in km/s for sample filtering.
        min_z_cut (float): Minimum redshift cut in km/s for sample filtering.

    This class enables sampling of redshifts for DLAs/LLSs based on the provided priors and parameters.

    Parameter prior (NHI, zLya) for strong Lya absorbers (logNHI = 17 - 23).

    This is a wrapper over the Monte Carlo samples already generated, so
    you should have already obtained the samples for the parameter prior
    somewhere else.

    We assume the same zLya prior as Garnett (2017):

        zLya ~ U(zmin, zmax)

        zmax = z_QSO - max_z_cut
        zmin = z_Ly∞ + min_z_cut
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        offset_samples: np.ndarray,
        log_nhi_samples: np.ndarray,
        max_z_cut: float = 3000,  # km/s
        min_z_cut: float = 3000,  # km/s
    ):
        super().__init__(params, prior)

        self._offset_samples = offset_samples
        self._log_nhi_samples = log_nhi_samples
        self._nhi_samples = 10**log_nhi_samples

    @property
    def offset_samples(self) -> np.ndarray:
        return self._offset_samples

    @property
    def log_nhi_samples(self) -> np.ndarray:
        return self._log_nhi_samples

    @property
    def nhi_samples(self) -> np.ndarray:
        return self._nhi_samples

    def sample_z_dlas(self, wavelengths: np.ndarray, z_qso: float) -> np.ndarray:
        sample_z_dlas = (
            self.min_z_dla(wavelengths, z_qso)
            + (
                self.params.max_z_dla(wavelengths, z_qso)
                - self.min_z_dla(wavelengths, z_qso)
            )
            * self._offset_samples
        )

        return sample_z_dlas

    def min_z_dla(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines minimum z_DLA to search

        We only consider z_dla within the modelling range.
        """
        rest_wavelengths = self.params.emitted_wavelengths(wavelengths, z_qso)
        ind = (rest_wavelengths >= self.params.min_lambda) & (
            rest_wavelengths <= self.params.max_lambda
        )

        # Here change to Lylimit minimum
        return np.max(
            [
                np.min(wavelengths[ind]) / self.params.lya_wavelength - 1,
                self.params.observed_wavelengths(self.params.lyman_limit, z_qso)
                / self.params.lya_wavelength
                - 1
                + self.params.min_z_cut,
            ]
        )

        # Here change to Lyb minimum
        # return np.max(
        #     [
        #         np.min(wavelengths[ind]) / self.params.lya_wavelength - 1,
        #         self.params.observed_wavelengths(self.params.lyb_wavelength, z_qso)
        #         / self.params.lya_wavelength
        #         - 1
        #         + self.params.min_z_cut,
        #     ]
        # )

    def _build_pdf(self):
        """
        Benchmark PDF we used in Garnett (2017), without mixing with a uniform prior
        """
        # directly use the fitted poly values in the Garnett (2017)
        unnormalized_dla_pdf = lambda log_nhi: (
            np.exp(-1.2695 * log_nhi**2 + 50.863 * log_nhi - 509.33)
        )
        # add a flat prior at the low column density end
        unnormalized_pdf = lambda log_nhi: (
            unnormalized_dla_pdf(log_nhi) * (log_nhi >= 20.03)
            + unnormalized_dla_pdf(20.03) * (log_nhi < 20.03)
        )

        Z = quad(unnormalized_pdf, 17.2, 23.0)[0]

        self.normalized_pdf = lambda log_nhi: unnormalized_pdf(log_nhi) / Z

    def pdf(self, log_nhi: float) -> float:
        """
        The logNHI pdf used in Garnett (2017) paper.
        """
        return self.normalized_pdf(log_nhi)


def get_prior_catalog(param: Parameters):
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
    param = Parameters(
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
    plt.figure(figsize=(16, 3))

    # Mean function
    plt.plot(
        gp.X,  # quasar spectrum's rest-frame wavelengths
        gp.Y,  # quasar spectrum's flux
        label="Data",
    )
    plt.fill_between(
        gp.X,
        gp.Y - 2 * np.sqrt(gp.v),
        gp.Y + 2 * np.sqrt(gp.v),
        label="SDSS Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )
    plt.plot(
        gp.rest_wavelengths,
        gp.mu,
        label="GP null model (mu = continuum)",
        color="C3",
        ls="--",
    )
    plt.plot(
        gp.X,
        gp.this_mu,
        label="GP null model (mu = meanflux)",  # GP model's mean function
    )
    plt.xlabel("Rest-frame Wavelengths [$\AA$]")
    plt.ylabel("Normalized Flux")
    plt.legend(loc="upper right")
    plt.ylim(-1, 5)
    plt.tight_layout()


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
    nth_lya: int,
    lya_gp: LLSGPDR12,
    gp: NullGPDR12,
    z_qso: float,
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
    # nth_lya = max_Lya  # here we plot all of the searches

    sample_z_dlas = lya_gp.dla_samples.sample_z_dlas(
        lya_gp.this_wavelengths, lya_gp.z_qso
    )

    # [color sequence] convert sample log likelihoods to values in (0, 1)
    sample_log_likelihoods = lya_gp.sample_log_likelihoods[
        :, 0
    ]  # only query the DLA(1) likelihoods
    # TODO: marginalize over k DLAs
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)

    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)

    # scale to make the colour more visible
    # TODO: make it more reasonable. scatter only takes values between [0, 1].
    colours = colours * 5 - 4
    colours[colours < 0] = 0

    # Canvas with two panels
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    # 1. Real spectrum space
    # N * (1~k models) * (1~k MAP dlas)
    MAP_z_dla, MAP_log_nhi = lya_gp.maximum_a_posteriori()
    # make them to be 1-D array
    map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
    map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]
    # feed in MAP values and get the absorption profile given (z_dlas, nhis)
    lya_mu, lya_M, lya_omega2 = lya_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

    # Only plot the spectrum within the search range
    this_rest_wavelengths = lya_gp.x
    ind = this_rest_wavelengths < lya_gp.params.lya_wavelength

    this_rest_wavelengths = this_rest_wavelengths[ind]
    lya_mu = lya_mu[ind]

    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
        lya_gp.Y[ind],
    )
    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
        lya_mu,
        label=r"$\mathcal{M}$"
        + r" HCD({n}); ".format(n=nth_lya)
        + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
        + "lognhi = ({})".format(",".join("{:.3g}".format(n) for n in map_log_nhis)),
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
    ax[1].scatter(
        sample_z_dlas,
        lya_gp.dla_samples.log_nhi_samples,
        c=colours,
        marker=".",
        alpha=0.25,
    )
    # MAP estimate
    ax[1].scatter(
        map_z_dlas,
        map_log_nhis,
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
    nth_lya: int,
    lya_gp: LLSGPDR12,
    gp: NullGPDR12,
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
):
    """
    Plots the extended spectrum prediction for a quasar, incorporating the effects of Lyman Alpha absorbers.

    Parameters:
        nth_lya (int): The index of the absorber to plot.
        lya_gp (LLSGPDR12): An instance of LLSGPDR12 representing the GP model with LLS.
        gp (NullGPDR12): An instance of NullGPDR12 representing the GP model without LLS.
        rest_wavelengths (np.ndarray): Array of rest wavelengths for the quasar spectrum.
        flux (np.ndarray): Array of flux values for the quasar spectrum.
    """
    # 1. Real spectrum space
    # N * (1~k models) * (1~k MAP dlas)
    MAP_z_dla, MAP_log_nhi = lya_gp.maximum_a_posteriori()
    # make them to be 1-D array
    map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
    map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]
    # feed in MAP values and get the absorption profile given (z_dlas, nhis)
    lya_mu, lya_M, lya_omega2 = lya_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

    # Only plot the spectrum within the search range
    this_rest_wavelengths = lya_gp.x
    ind = this_rest_wavelengths < lya_gp.params.lya_wavelength

    this_rest_wavelengths = this_rest_wavelengths[ind]
    lya_mu = lya_mu[ind]

    fig, ax = plt.subplots(1, 1, figsize=(16, 3))

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
        lya_mu,
        label=r"$\mathcal{M}$"
        + r" HCD({n}); ".format(n=nth_lya)
        + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
        + "lognhi = ({})".format(",".join("{:.3g}".format(n) for n in map_log_nhis)),
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
    ax.set_xlim(750, 1415)
    ax.set_ylim(-1, 5)
    plt.legend()
    plt.xlabel("Rest-frame Wavelengths [$\AA$]")
    plt.ylabel("Normalized Flux")
    plt.tight_layout()

    return fig, ax


########################### Plotting End #######################################


def save_processed_file(
    filename: str,
    gp: NullGPDR12,
    lya_gp: LLSGPDR12,
    log_likelihoods_lya: np.ndarray,
    z_qso: float,
    max_Lya: int,
):
    """
    Saves the processed data from the GP models and Lya absorber analysis into an HDF5 file.

    Parameters:
        filename (str): The path to the file where the data will be saved.
        gp (NullGPDR12): An instance of NullGPDR12 representing the GP model without LLS.
        lya_gp (LLSGPDR12): An instance of LLSGPDR12 representing the GP model with LLS.
        log_likelihoods_lya (np.ndarray): Log likelihoods of the Lya absorber models.
        z_qso (float): The redshift of the quasar.
        max_Lya (int): The maximum number of Lyman Alpha absorbers considered in the analysis.
    """
    # Calculate posteriors for the null model and Lya models.

    log_posteriors_lya = log_likelihoods_lya + lya_gp.log_priors(z_qso, max_Lya)
    log_posterior_null = gp.log_model_evidence() + gp.log_prior(z_qso)

    log_posteriors = np.append(log_posterior_null, log_posteriors_lya)

    max_log_posterior = log_posteriors.max()

    model_posteriors = np.exp(log_posteriors - max_log_posterior)
    model_posteriors = model_posteriors / model_posteriors.sum()

    MAP_z_lyas, MAP_log_nhis = lya_gp.maximum_a_posteriori()

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

        f.create_dataset(
            "sample_log_likelihoods_lya", data=lya_gp.sample_log_likelihoods
        )
        f.create_dataset("base_sample_inds", data=lya_gp.base_sample_inds)

        f.create_dataset("log_prior_no_lya", data=gp.log_prior(z_qso))
        f.create_dataset("log_priors_lya", data=lya_gp.log_priors(z_qso, max_Lya))

        f.create_dataset("log_likelihood_no_lya", data=gp.log_model_evidence())
        f.create_dataset("log_likelihoods_lya", data=log_likelihoods_lya)

        f.create_dataset("log_posterior_null", data=log_posterior_null)
        f.create_dataset("log_posteriors_lya", data=log_posteriors_lya)

        f.create_dataset("MAP_z_lyas", data=MAP_z_lyas)
        f.create_dataset("MAP_log_nhis", data=MAP_log_nhis)

        f.create_dataset("model_posteriors", data=model_posteriors)

        # also save zQSOs
        f.create_dataset("z_qsos", data=np.array([z_qso]))

    return model_posteriors


########################### Main Sampling Function ######################################
def main(
    nspec: int,
    max_Lya: int = 4,
    num_lines: int = 4,
    img_dir: str = "images-lls/",
    lls_sample_h5: str = "../data/dr12q/processed/lls_samples.h5",
):
    """
    Main function to execute the analysis pipeline for detecting LLS and DLA in quasar spectra.

    Parameters:
        nspec (int): Index of the spectrum to analyze.
        max_Lya (int): Maximum number of Lyman Alpha absorbers to search for.
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

    # Save the Fagamalli quasar spectrum information into readable text file
    print(
        "[Info] Saving the Fagamalli quasar spectrum information into readable text file ..."
    )
    # print out the Fagamaili quasar spectrum information
    print("Quasar Name: {}".format(loader.quasar_name[nspec]))
    print("Redshift: {}".format(z_qso))
    print("SN_1150A: {}".format(loader.SN_1150A[nspec]))
    print("Classification Outcome: {}".format(loader.classification_outcome[nspec]))
    print("Note:  (1: quasar with LLS; 2: quasar without LLS; 4: non quasar)")
    with open(os.path.join(img_dir, "Fagamalli_quasar_info.txt"), "w") as f:
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
        f.write("Note:  (1: quasar with LLS; 2: quasar without LLS; 4: non quasar)")
        f.write("LLS Redshift: {}\n".format(loader.LLS_redshift[nspec]))

    # If we haven't downloaded the file, this cell will help you download the file from SDSS database
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
            retrieve_raw_spec(int(plate), int(mjd), int(fiber_id), release="dr14q")
        except Exception as e:
            print("Error: ", e)
            print("Downloading from dr14q ...")
            retrieve_raw_spec(int(plate), int(mjd), int(fiber_id), release="dr12q")

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
    with h5py.File(lls_sample_h5, "r") as f:
        halton_sequence = f["halton_sequence"][()]
        samples_log_nhis = f["samples_log_nhis"][()]

    # LyaSamples: this is the prior of absorber redshifts and column densities
    # p(z_abs) p(NHI)
    lya_samples = LyaSamples(
        params=param,
        prior=prior,
        # offset samples
        offset_samples=halton_sequence[:, 1],
        log_nhi_samples=samples_log_nhis,
        max_z_cut=3000,
        min_z_cut=3000,
    )
    plt.hist(samples_log_nhis, bins=100, density=True, label="Monte Carlo samples")
    plt.xlabel("log NHI")
    plt.ylabel("Probability density function")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "monte_carlo_samples.png"), dpi=150, format="png")
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
        min_z_separation=2000,  # 2. you have the minimum zabs separateion, in unit of km/s
        learned_file="data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        broadening=True,
        prev_tau_0=0.0023,
        prev_beta=3.65,
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
    log_likelihoods_lya = lya_gp.log_model_evidences(max_Lya)

    print("Spend {:.4g} seconds".format(time.time() - tt))

    # Save the processed data into an HDF5 file
    print("[Info] Saving the processed data into an HDF5 file ...")
    model_posteriors = save_processed_file(
        os.path.join(img_dir, "processed.h5"),
        gp,
        lya_gp,
        log_likelihoods_lya,
        z_qso,
        max_Lya,
    )

    MAP_z_lyas, MAP_log_nhis = lya_gp.maximum_a_posteriori()

    # MAP estimates
    nth_lya = model_posteriors.argmax()  # the index of the maximum posterior
    # Plot the sample predictions
    print("[Info] Plotting the sample predictions ...")
    plot_sample_predictions(
        nth_lya,
        lya_gp,
        gp,
        z_qso,
    )
    plt.savefig(os.path.join(img_dir, "sample_predictions.png"), dpi=150, format="png")
    plt.clf()
    plt.close()

    # Plot the extended spectrum prediction
    plot_prediction_extended_spectrum(
        nth_lya,
        lya_gp,
        gp,
        rest_wavelengths,
        flux,
    )
    plt.savefig(
        os.path.join(img_dir, "extended_predictions.png"), dpi=150, format="png"
    )
    plt.clf()
    plt.close()

    # Save the basic information of the LLs detection run:
    with open(os.path.join(img_dir, "LLS_detection_info.txt"), "w") as f:
        f.write("Spectrum: {}\n".format(filename))
        f.write("Max Lya Absorbers: {}\n".format(max_Lya))
        f.write("Number of Lya Series Lines: {}\n".format(num_lines))
        f.write("LLS Sample H5: {}\n".format(lls_sample_h5))
        f.write("Processed File: {}\n".format(os.path.join(img_dir, "processed.h5")))
        # Number of LLS samples used
        f.write("Number of LLS Samples: {}\n".format(len(samples_log_nhis)))

        # Detected number of absorbers and model posteriors, and the redshifts and nhis
        f.write("Detected Number of Absorbers: {}\n".format(nth_lya))
        f.write("Model Posteriors: {}\n".format(model_posteriors))
        f.write("MAP z_lyas: {}\n".format(MAP_z_lyas))
        f.write("MAP log_nhis: {}\n".format(MAP_log_nhis))

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
        "--max_Lya",
        type=int,
        default=4,
        help="The maximum number of Lya absorbers to search.",
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
    args = parser.parse_args()

    main(args.nspec, args.max_Lya, args.num_lines, args.img_dir, args.lls_sample_h5)
    # example: python examples/gp_find_lls.py --nspec 0 --max_Lya 4 --num_lines 4 --img_dir images-lls/ --lls_sample_h5 data/dr12q/processed/lls_samples.h5
