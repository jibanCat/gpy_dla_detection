"""
A GP class for running MCMC CIV model using MCMC in a given sightline.
"""
from typing import Tuple, Optional

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
import h5py

from scipy import interpolate

from .civ_set_parameter import CIVParameters as Parameters # use Zesimation model for now until Reza gives me the model
from .model_priors import PriorCatalog
from .null_gp import NullGP
from .voigt_civ import voigt_absorption

# this could be replaced to DLASamples in the future;
# I import this is for the convenient of my autocomplete
# from .dla_samples import DLASamplesMAT

# the mcmc log posterior function
import emcee
from .civ_log_posterior_mcmc import log_posterior


class CIVGP(NullGP):
    """
    CIV GP model for QSO emission + CIV intervening:
        p(y | λ, σ², M, z_civ, logNCIV)

    additional two parameters (z_civ, logNCIV) will control the position
    and the strength of the absorption intervening on the QSO emission.

    Since the integration is not tractable, so we use QMC to approximate
    the model evidence. 

    How many QMC samples will be defined in Parameters and DLASamples.

    :param rest_wavelengths: λ, the range of λ you model your GP on QSO emission
    :param mu: mu, the mean model of the GP.
    :param M: M, the low rank decomposition of the covariance kernel: K = MM^T.
    :param broadening: whether to implement the SDSS instrumental broadening in the Voigt profile,
        default is True.

    ..Note: MCMC embedded in the class as an instance method.
    """

    def __init__(
        self,
        params: Parameters,
        # prior: PriorCatalog,
        # dla_samples: DLASamplesMAT,
        rest_wavelengths: np.ndarray,
        mu: np.ndarray,
        M: np.ndarray,
        min_z_separation: float = 3000.0,
        broadening: bool = True,
    ):


        self.params = params
        # self.z_qso_samples = z_qso_samples

        # learned GP model
        self.rest_wavelengths = rest_wavelengths
        self.mu = mu
        self.M = M

        # preprocess model interpolants
        self.mu_interpolator = interpolate.interp1d(rest_wavelengths, mu)
        self.list_M_interpolators = [
            interpolate.interp1d(rest_wavelengths, eigenvector) for eigenvector in M.T
        ]

        self.min_z_separation = self.params.kms_to_z(min_z_separation)


        self.broadening = broadening

    def run_mcmc(
        self,
        nwalkers: int,
        kth_civ: int = 1,
        nsamples: int = 5000,
        pos: Optional[np.ndarray] = None,
        skip_initial_state_check:bool=True,
    ) -> emcee.EnsembleSampler:
        """
        An MCMC implementation for marginalizing log likelihood at kth DLA model.

        MCMC should give a more accurate parameter estimation than maximum a
        posteriori on the QMC samples.
        """
        # get the prior range for the zDLA
        min_z_civ = self.params.min_z_civ(self.this_wavelengths, self.z_qso)
        max_z_civ = self.params.max_z_civ(self.this_wavelengths, self.z_qso)

        # prior range for logNCIV
        # min_log_nhi = self.dla_samples.uniform_min_log_nhi
        # max_log_nhi = self.dla_samples.uniform_max_log_nhi
        # TODO: modulize properly
        min_log_nciv = 12.88
        max_log_nciv = 14.5

        min_sigma = 1e6 # cm/s
        max_sigma = 12e6 # cm/s

        # make the pdf function here
        # uniform component of column density prior
        u = stats.uniform(loc=min_log_nciv,
            scale=max_log_nciv - min_log_nciv)

        # directly use the fitted poly values in the Garnett (2017)
        # unnormalized_pdf = lambda nhi: (np.exp(
        #     -1.2695 * nhi**2 + 50.863 * nhi -509.33
        # ))
        # Z = quad(unnormalized_pdf, self.dla_samples.fit_min_log_nhi, 25.0)[0] # hard-coded 25.0

        # create the PDF of the mixture between the unifrom distribution and
        # the distribution fit to the data
        normalized_pdf = lambda nciv: u.pdf(nciv)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            kth_civ * 3,
            log_posterior,
            args=(
                self.this_wavelengths,
                self.y,
                self.v,
                self.z_qso,
                min_z_civ,
                max_z_civ,
                min_log_nciv,
                max_log_nciv,
                min_sigma,
                max_sigma,
                normalized_pdf,
                self.padded_wavelengths,
                self.this_mu,
                self.this_M,
                self.pixel_mask,
                self.ind_unmasked,
                2, # num lines: 2, doublet
            ),
        )

        # initial position
        if pos is None:
            pos = np.concatenate(
                [
                    np.random.uniform(low=min_z_civ, high=max_z_civ,  size=nwalkers)[:, None],
                    np.random.uniform(low=min_log_nciv, high=max_log_nciv, size=nwalkers)[:, None],
                    np.random.uniform(low=min_sigma, high=max_sigma, size=nwalkers)[:, None],
                ],
                axis=1,
            )

            assert pos.shape[0] == nwalkers

        sampler.run_mcmc(pos, nsamples, progress=True, skip_initial_state_check=skip_initial_state_check)

        # return the sampler after the sampling
        return sampler

    def get_interp(
        self, x: np.ndarray, y: np.ndarray, wavelengths: np.ndarray, z_qso: float
    ) -> None:
        """
        Build and interpolate the GP model onto the observed data.
        
        p(y | λ, zqso, v, ω, M_nociv) = N(y; μ, K + V)
        
        :param x: this_rest_wavelengths
        :param y: this_flux
        :param wavelengths: observed wavelengths, put this arg is just for making the
            code looks similar to the MATLAB code. This could be taken off in the future.
        :param z_qso: quasar redshift

        Note: assume already pixel masked.
        """
        # interpolate model onto given wavelengths
        this_mu = self.mu_interpolator(x)
        this_M = self.M_interpolator(x)

        assert this_M.shape[1] == self.params.k

        # assign to instance attrs
        self.this_mu = this_mu
        self.this_M = this_M


    def this_civ_gp(
        self,
        z_civ: np.ndarray,
        nciv: np.ndarray,
        sigma: np.ndarray,
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
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # absorption corresponding to this sample
        absorption = voigt_absorption(
            self.padded_wavelengths, z_civ=z_civ[0], nciv=nciv[0], sigma=sigma[0], num_lines=self.params.num_lines,
        )

        # absorption corresponding to other CIVs in multiple CIV samples
        for j in range(1, k_civs):
            absorption = absorption * voigt_absorption(
                self.padded_wavelengths, z_civ=z_civ[j], nciv=nciv[j], sigma=sigma[j], num_lines=self.params.num_lines,
            )

        absorption = absorption[mask_ind]

        assert len(absorption) == len(self.this_mu)

        civ_mu = self.this_mu * absorption
        civ_M = self.this_M * absorption[:, None]

        return civ_mu, civ_M


class CIVGPMAT(CIVGP):
    """
    Load learned model from .mat file
    """

    def __init__(
        self,
        params: Parameters,
        # prior: PriorCatalog,
        # dla_samples: DLASamplesMAT,
        min_z_separation: float = 3000.0,
        learned_file: str = "data/dr7q/learned_model-C13_full.mat",
        broadening: bool = True,
    ):
        with h5py.File(learned_file, "r") as learned:

            rest_wavelengths = learned["rest_wavelengths"][:, 0]
            mu = learned["mu"][:, 0]
            M = learned["M"][()].T

        super().__init__(
            params,
            rest_wavelengths,
            mu,
            M,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )
