"""
A GP class for search Lyman limit systems (LLS) in quasar spectra,
including the alternative metal absorption lines (MgII, CIV).
"""
from typing import Tuple, Optional

import itertools
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
import h5py

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .dla_gp import DLAGP
from .voigt_lls import voigt_absorption as voigt_absorption_lls
from .voigt_civ import voigt_absorption as voigt_absorption_civ
from .voigt_mgii import voigt_absorption as voigt_absorption_mgii

from .dla_samples import DLASamples, DLASamplesMAT

from .cddf_samples import dla_normalized_pdf, civ_normalized_pdf, mgii_normalized_pdf


class LLSParameters(Parameters):
    civ_1548_wavelength = 1548.2040  # CIV transition wavelength  Å
    civ_1550_wavelength = 1550.77810  # CIV transition wavelengt
    mgii_2796_wavelength = 2796.3542699  # MgII transition wavelength  Å
    mgii_2803_wavelength = 2803.5314853  # MgII transition wavelength  Å

    def __init__(
        self,
        num_dla_samples: int = 100000,
        max_z_cut: float = 5000.0,  # max z_DLA = z_QSO - max_z_cut
        min_z_cut: float = 5000.0,  # min z_DLA = z_Ly∞ + min_z_cut
        **kwargs  # number of parameter samples
    ):
        super().__init__(
            num_dla_samples=num_dla_samples,
            max_z_cut=max_z_cut,
            min_z_cut=min_z_cut,
            **kwargs
        )

    def min_z_dla(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines minimum z_DLA to search

        We only consider z_dla within the modelling range.
        """
        rest_wavelengths = self.emitted_wavelengths(wavelengths, z_qso)
        ind = (rest_wavelengths >= self.min_lambda) & (
            rest_wavelengths <= self.max_lambda
        )

        # Here change to Lylimit minimum
        return np.max(
            [
                np.min(wavelengths[ind]) / self.lya_wavelength - 1,
                self.observed_wavelengths(self.lyman_limit, z_qso) / self.lya_wavelength
                - 1
                + self.min_z_cut,
            ]
        )

    def min_z_civ(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines minimum z_CIV to search.

        We only consider z_civ overlapping with the HI absorber search range.
        """
        z_hi = self.min_z_dla(wavelengths, z_qso)
        return (z_hi + 1) / self.civ_1550_wavelength * self.lya_wavelength - 1

    def max_z_civ(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines maximum z_CIV to search.

        We only consider z_civ overlapping with the HI absorber search range.
        """
        z_hi = self.max_z_dla(wavelengths, z_qso)
        return (z_hi + 1) / self.civ_1550_wavelength * self.lya_wavelength - 1

    def min_z_mgii(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines minimum z_MgII to search.

        We only consider z_mgii overlapping with the HI absorber search range.
        """
        z_hi = self.min_z_dla(wavelengths, z_qso)
        return (z_hi + 1) / self.mgii_2803_wavelength * self.lya_wavelength - 1

    def max_z_mgii(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines maximum z_MgII to search.

        We only consider z_mgii overlapping with the HI absorber search range.
        """
        z_hi = self.max_z_dla(wavelengths, z_qso)
        return (z_hi + 1) / self.mgii_2803_wavelength * self.lya_wavelength - 1


class LyaSamples(DLASamples):
    """
    Parameters
    ----------
    params : Parameters
        Global parameters used for model training and defining priors.
    prior : PriorCatalog
        Prior catalog based on the SDSS DR9 Lyman Alpha catalog.
    offset_samples : np.ndarray
        Monte Carlo samples of prior distributions for zLya.
    log_nhi_samples : np.ndarray
        Monte Carlo samples of prior distributions for NHI.
    max_z_cut : float
        Maximum redshift cut in km/s.
    min_z_cut : float
        Minimum redshift cut in km/s.

    This class enables sampling of redshifts for DLAs/LLSs based on the provided priors and parameters.

    Parameter prior (NHI, zLya) for strong Lya absorbers (logNHI = 17.5 - 23).

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
        params: LLSParameters,
        prior: PriorCatalog,
        offset_samples: np.ndarray,
        log_nhi_samples: np.ndarray,
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
            self.params.min_z_dla(wavelengths, z_qso)
            + (
                self.params.max_z_dla(wavelengths, z_qso)
                - self.params.min_z_dla(wavelengths, z_qso)
            )
            * self._offset_samples
        )

        return sample_z_dlas

    def pdf(self, log_nhi: float) -> float:
        """
        The logNHI pdf from CDDF.
        """

        return dla_normalized_pdf(log_nhi)


class CIVSamples(DLASamples):
    """
    Parameters
    ----------
    params : Parameters
        Global parameters used for model training and defining priors.
    prior : PriorCatalog
        Prior catalog based on the SDSS DR9 Lyman Alpha catalog.
    offset_samples : np.ndarray
        Monte Carlo samples of prior distributions for zCIV.
    log_nciv_samples : np.ndarray
        Monte Carlo samples of prior distributions for NCIV.
    max_z_cut : float
        Maximum redshift cut in km/s.
    min_z_cut : float
        Minimum redshift cut in km/s.

    This class enables sampling of redshifts for CIV absorbers based on the provided priors and parameters.

    Parameter prior (NCIV, zCIV) for CIV absorbers (logNCIV = 12.5 - 15).

    This is a wrapper over the Monte Carlo samples already generated, so
    you should have already obtained the samples for the parameter prior
    somewhere else.

    We assume the same zCIV prior as Garnett (2017), but with a conversion from HI to metals.

        zCIV ~ U(zmin, zmax)

        zmax = (z_QSO - max_z_cut + 1) / 1549 * 1216 - 1
        zmin = (z_Ly∞ + min_z_cut + 1) / 1549 * 1216 - 1
    """

    def __init__(
        self,
        params: LLSParameters,
        prior: PriorCatalog,
        offset_samples: np.ndarray,
        log_civ_samples: np.ndarray,
    ):
        super().__init__(params, prior)

        self._offset_samples = offset_samples
        self._log_nciv_samples = log_civ_samples
        self._nciv_samples = 10**log_civ_samples

    @property
    def offset_samples(self) -> np.ndarray:
        return self._offset_samples

    @property
    def log_nciv_samples(self) -> np.ndarray:
        return self._log_nciv_samples

    @property
    def nciv_samples(self) -> np.ndarray:
        return self._nciv_samples

    def sample_z_civs(self, wavelengths: np.ndarray, z_qso: float) -> np.ndarray:
        """
        Sample z_CIV from the prior distribution.

        Parameters:
        - wavelengths: An array of wavelengths in the quasar spectra.
        - z_qso: The redshift of the quasar.

        Returns:
        - sample_z_civs: An array of sampled redshifts for CIV absorbers.
        """
        sample_z_civs = (
            self.params.min_z_civ(wavelengths, z_qso)
            + (
                self.params.max_z_civ(wavelengths, z_qso)
                - self.params.min_z_civ(wavelengths, z_qso)
            )
            * self._offset_samples
        )

        return sample_z_civs

    def pdf(self, log_nciv: float) -> float:
        """
        The logNCIV pdf from CDDF.
        """

        return civ_normalized_pdf(log_nciv)


class MgIISamples(DLASamples):
    """
    Parameters
    ----------
    params : Parameters
        Global parameters used for model training and defining priors.
    prior : PriorCatalog
        Prior catalog based on the SDSS DR9 Lyman Alpha catalog.
    offset_samples : np.ndarray
        Monte Carlo samples of prior distributions for zMgII.
    log_nmgii_samples : np.ndarray
        Monte Carlo samples of prior distributions for NMgII.
    max_z_cut : float
        Maximum redshift cut in km/s.
    min_z_cut : float
        Minimum redshift cut in km/s.

    This class enables sampling of redshifts for MgII absorbers based on the provided priors and parameters.

    Parameter prior (NMgII, zMgII) for MgII absorbers (logNMgII = 12.5 - 15).

    This is a wrapper over the Monte Carlo samples already generated, so
    you should have already obtained the samples for the parameter prior
    somewhere else.

    We assume the same zMgII prior as Garnett (2017), but with a conversion from HI to metals.

        zMgII ~ U(zmin, zmax)

        zmax = (z_QSO - max_z_cut + 1) / 2796 * 1216 - 1
        zmin = (z_Ly∞ + min_z_cut + 1) / 2796 * 1216 - 1
    """

    def __init__(
        self,
        params: LLSParameters,
        prior: PriorCatalog,
        offset_samples: np.ndarray,
        log_mgii_samples: np.ndarray,
    ):
        super().__init__(params, prior)

        self._offset_samples = offset_samples
        self._log_nmgii_samples = log_mgii_samples
        self._nmgii_samples = 10**log_mgii_samples

    @property
    def offset_samples(self) -> np.ndarray:
        return self._offset_samples

    @property
    def log_nmgii_samples(self) -> np.ndarray:
        return self._log_nmgii_samples

    @property
    def nmgii_samples(self) -> np.ndarray:
        return self._nmgii_samples

    def sample_z_mgiis(self, wavelengths: np.ndarray, z_qso: float) -> np.ndarray:
        """
        Sample z_MgII from the prior distribution.

        Parameters:
        - wavelengths: An array of wavelengths in the quasar spectra.
        - z_qso: The redshift of the quasar.

        Returns:
        - sample_z_mgiis: An array of sampled redshifts for MgII absorbers.
        """
        sample_z_mgiis = (
            self.params.min_z_mgii(wavelengths, z_qso)
            + (
                self.params.max_z_mgii(wavelengths, z_qso)
                - self.params.min_z_mgii(wavelengths, z_qso)
            )
            * self._offset_samples
        )

        return sample_z_mgiis

    def pdf(self, log_nmgii: float) -> float:
        """
        The logNMgII pdf from CDDF.
        """

        return mgii_normalized_pdf(log_nmgii)


# A model to search LLS with alternatives of CIV and MgII
class LLSGPDR12(DLAGP):
    """
    This class extends the DLAGP class to model the presence of HI absorbers within the spectra.
    It incorporates the Voigt profile parameterization for HI absorbers, as well as the alternative
    metal absorption lines (CIV and MgII).

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
        params: LLSParameters,
        prior: PriorCatalog,
        lya_samples: LyaSamples,
        civ_samples: CIVSamples,
        mgii_samples: MgIISamples,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        prev_tau_0: float = 0.00554,
        prev_beta: float = 3.182,
        min_z_separation: float = 2000.0,  # km/s
        broadening: bool = True,
    ):
        # Load the learned model parameters from the specified file
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

        self.civ_samples = civ_samples
        self.mgii_samples = mgii_samples

        # cache for absorption profiles
        self.absorption_cache = {}
        # integer label for the absorption profile
        self.lls_label = 0
        self.civ_label = 1
        self.mgii_label = 2

    def voigt_absorption_lls_precomputed(self, z_lls: float, nhis: float) -> np.ndarray:
        """
        Quickly query the precomputed absorption profile for a given set of parameters,
        without the need for re-computation.

        Will be useful for multiple absorption cases, where we re-use the same absorption profile.

        Parameters:
        - z_lls: The redshift of the LLS absorber.
        - nhis: The column density of the LLS absorber.

        Returns:
        - absorption: The absorption profile for the given parameters.
        """
        # check if the absorption profile is already computed
        if (self.lls_label, z_lls, nhis) in self.absorption_cache:
            print(
                "[Debug] Query the cache for LLS absorption profile",
                (self.lls_label, z_lls, nhis),
            )
            return self.absorption_cache[(self.lls_label, z_lls, nhis)]

        # to retain only unmasked pixels from computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # [broadening] use the padded wavelengths for convolution
        # otherwise, should use unmasked wavelengths.
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        # compute the absorption profile
        absorption = voigt_absorption_lls(
            wavelengths,
            z_lls=z_lls,
            nhi=nhis,
            num_lines=self.params.num_lines,
            broadening=self.broadening,
        )

        absorption = absorption[mask_ind]

        # cache the absorption profile
        self.absorption_cache[(self.lls_label, z_lls, nhis)] = absorption

        return absorption

    def voigt_absorption_civ_precomputed(self, z_civ: float, nciv: float) -> np.ndarray:
        """
        Quickly query the precomputed absorption profile for a given set of parameters,
        without the need for re-computation.

        Will be useful for multiple absorption cases, where we re-use the same absorption profile.

        Parameters:
        - z_civ: The redshift of the CIV absorber.
        - nciv: The column density of the CIV absorber.

        Returns:
        - absorption: The absorption profile for the given parameters.
        """
        # check if the absorption profile is already computed
        if (self.civ_label, z_civ, nciv) in self.absorption_cache:
            return self.absorption_cache[(self.civ_label, z_civ, nciv)]

        # to retain only unmasked pixels from computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # [broadening] use the padded wavelengths for convolution
        # otherwise, should use unmasked wavelengths.
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        # compute the absorption profile
        absorption = voigt_absorption_civ(
            wavelengths,
            z_civ=z_civ,
            nciv=nciv,
            sigma=np.sqrt(2)
            * 10.5e5,  # km/s, from Kim et al. 2003, median b = 10.5 km/s
            broadening=self.broadening,
        )

        absorption = absorption[mask_ind]

        # cache the absorption profile
        self.absorption_cache[(self.civ_label, z_civ, nciv)] = absorption

        return absorption

    def voigt_absorption_mgii_precomputed(
        self, z_mgii: float, nmgii: float
    ) -> np.ndarray:
        """
        Quickly query the precomputed absorption profile for a given set of parameters,
        without the need for re-computation.

        Will be useful for multiple absorption cases, where we re-use the same absorption profile.

        Parameters:
        - z_mgii: The redshift of the MgII absorber.
        - nmgii: The column density of the MgII absorber.

        Returns:
        - absorption: The absorption profile for the given parameters.
        """
        # check if the absorption profile is already computed
        if (self.mgii_label, z_mgii, nmgii) in self.absorption_cache:
            return self.absorption_cache[(self.mgii_label, z_mgii, nmgii)]

        # to retain only unmasked pixels from computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # [broadening] use the padded wavelengths for convolution
        # otherwise, should use unmasked wavelengths.
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        # compute the absorption profile
        absorption = voigt_absorption_mgii(
            wavelengths,
            z_mgii=z_mgii,
            nmgii=nmgii,
            sigma=np.sqrt(2)
            * 5.7e5,  # km/s, from Churchill et al. 2020, median b = 5.7 km/s
            num_lines=2,  # doublet
            broadening=self.broadening,
        )

        absorption = absorption[mask_ind]

        # cache the absorption profile
        self.absorption_cache[(self.mgii_label, z_mgii, nmgii)] = absorption

        return absorption

    def this_lls_gp(
        self, z_lls: np.ndarray, nhis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the HI GP model with k intervening HI profiles onto
        the mean and covariance.
        Only return the absorption profile.

        Parameters:
        - z_lls: An array of redshifts for the intervening HI absorbers.
        - nhis: An array of column densities for the intervening HI absorbers.

        Returns:
        - absorption: The absorption profile for the given parameters.
        """
        assert len(z_lls) == len(nhis)

        k_lls = len(z_lls)

        # [Zero absorption] if no LLS is present, return flux = 1 with length equal to the number of wavelengths
        if k_lls == 0:
            return np.ones_like(self.this_mu)

        # HI absorption corresponding to this sample
        absorption = self.voigt_absorption_lls_precomputed(
            z_lls=z_lls[0],
            nhis=nhis[0],
        )

        # multiple HI absorptions
        for j in range(1, k_lls):
            absorption = absorption * self.voigt_absorption_lls_precomputed(
                z_lls=z_lls[j],
                nhis=nhis[j],
            )

        assert len(absorption) == len(self.this_mu)

        return absorption

    def this_mgii_gp(
        self, z_mgiis: np.ndarray, nmgii: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the GP model with k intervening MgII profiles onto
        the mean and covariance.

        Parameters:
        - z_mgiis: An array of redshifts for the intervening MgII absorbers.
        - nmgii: An array of column densities for the intervening MgII absorbers.

        Returns:
        - absorption: The absorption profile for the given parameters.
        """
        # same as lls case instead using voigt_absorption_mgii
        assert len(z_mgiis) == len(nmgii)

        k_mgiis = len(z_mgiis)

        # [Zero absorption] if no MgII is present, return flux = 1 with length equal to the number of wavelengths
        if k_mgiis == 0:
            return np.ones_like(self.this_mu)

        # MgII absorption corresponding to this sample
        absorption = self.voigt_absorption_mgii_precomputed(
            z_mgii=z_mgiis[0],
            nmgii=nmgii[0],
        )

        # multiple MgII absorptions
        for j in range(1, k_mgiis):
            absorption = absorption * self.voigt_absorption_mgii_precomputed(
                z_mgii=z_mgiis[j],
                nmgii=nmgii[j],
            )

        assert len(absorption) == len(self.this_mu)

        return absorption

    def this_civ_gp(
        self, z_civs: np.ndarray, ncivs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the GP model with k intervening CIV profiles onto
        the mean and covariance.

        Parameters:
        - z_civs: An array of redshifts for the intervening CIV absorbers.
        - ncivs: An array of column densities for the intervening CIV absorbers.

        Returns:
        - absorption: The absorption profile for the given parameters.
        """
        # same as lls case instead using voigt_absorption_civ
        assert len(z_civs) == len(ncivs)

        k_civs = len(z_civs)

        # [Zero absorption] if no CIV is present, return flux = 1 with length equal to the number of wavelengths
        if k_civs == 0:
            return np.ones_like(self.this_mu)

        # CIV absorption corresponding to this sample
        absorption = self.voigt_absorption_civ_precomputed(
            z_civ=z_civs[0],
            nciv=ncivs[0],
        )

        # multiple CIV absorptions
        for j in range(1, k_civs):
            absorption = absorption * self.voigt_absorption_civ_precomputed(
                z_civ=z_civs[j],
                nciv=ncivs[j],
            )

        assert len(absorption) == len(self.this_mu)

        return absorption

    def sample_log_likelihood_k_llss(
        self,
        z_llss: np.ndarray,
        nhis: np.ndarray,
        z_mgiis: np.ndarray,
        nmgii: np.ndarray,
        z_civs: np.ndarray,
        ncivs: np.ndarray,
    ) -> float:
        """
        Compute the log likelihood of k LLSs, l MgII, and m CIV absorbers,
        conditioned on the GP model.

        The likelihood can be expressed as:
                p(D | z_QSO, k LLSs, l MgII, m CIV)

        Parameters:
        - z_llss: An array of redshifts for the intervening LLS absorbers.
        - nhis: An array of column densities for the intervening LLS absorbers.
        - z_mgiis: An array of redshifts for the intervening MgII absorbers.
        - nmgii: An array of column densities for the intervening MgII absorbers.
        - z_civs: An array of redshifts for the intervening CIV absorbers.
        - ncivs: An array of column densities for the intervening CIV absorbers.

        Returns:
        - sample_log_likelihood: The log likelihood of the given absorbers conditioned on the GP model.
        """
        assert len(z_llss) == len(nhis)
        assert len(z_mgiis) == len(nmgii)
        assert len(z_civs) == len(ncivs)

        # the absorption profile from each absorber, check the length first
        abs_lls = self.this_lls_gp(z_llss, nhis)
        abs_mgii = self.this_mgii_gp(z_mgiis, nmgii)
        abs_civ = self.this_civ_gp(z_civs, ncivs)
        # use the combined absorption profile to multiply to the GP model
        abs_combined = abs_lls * abs_mgii * abs_civ

        abs_mu = self.this_mu * abs_combined
        abs_M = self.this_M * abs_combined[:, None]
        abs_omega2 = self.this_omega2 * abs_combined**2

        sample_log_likelihood = self.log_mvnpdf_low_rank(
            self.y, abs_mu, abs_M, abs_omega2 + self.v
        )

        return sample_log_likelihood

    def log_model_evidence_lls(self, max_lls: int) -> np.ndarray:
        """
        Compute the log model evidence for the presence of LLSs in the quasar spectra.
        """
        for num_lls in range(1, max_lls + 1):  # count from zero to max_lls
            # [Need to be parallelized]
            # Roman's code has this part to be parallelized.
            # TODO: threading this part
            for i in range(self.params.num_dla_samples):
                # query the 1st LLS parameter {z_lls, logNHI}_{i=1}
                z_lls = np.array([self.sample_z_dlas[i]])
                log_nhis = np.array([self.dla_samples.log_nhi_samples[i]])
                nhis = np.array([self.dla_samples.nhi_samples[i]])

                # query the 2:k LLS parameters {z_lls, logNHI}_{i=2}^k_lls
                if num_lls > 1:
                    base_ind = self.base_sample_inds_lls[: (num_lls - 1), i]

                    z_lls_2_k = self.sample_z_dlas[base_ind]
                    log_nhis_2_k = self.dla_samples.log_nhi_samples[base_ind]
                    nhis_2_k = self.dla_samples.nhi_samples[base_ind]

                    # append to samples to be applied on calculating the log likelihood
                    z_lls = np.append(z_lls, z_lls_2_k)
                    log_nhis = np.append(log_nhis, log_nhis_2_k)
                    nhis = np.append(nhis, nhis_2_k)

                    del z_lls_2_k, log_nhis_2_k, nhis_2_k

                # store the sample log likelihoods conditioned on k-LLSs solely
                self.sample_log_likelihoods[
                    i, num_lls, 0, 0
                ] = self.sample_log_likelihood_k_llss(
                    z_lls, nhis, np.array([]), np.array([]), np.array([]), np.array([])
                ) - np.log(
                    self.params.num_dla_samples
                )  # additional occams razor

            # check if any pair of absorbers in this sample is too close this has to
            # happen outside the parfor because "continue" slows things down
            # dramatically
            if num_lls > 1:
                # all_z_llss : (num_llss, num_lls_samples)
                ind = self.base_sample_inds_lls[
                    : (num_lls - 1), :
                ]  # (num_llss - 1, num_lls_samples)

                all_z_dlas = np.concatenate(
                    [self.sample_z_dlas[None, :], self.sample_z_dlas[ind]], axis=0
                )  # (num_dlas, num_dla_samples)

                ind = np.any(
                    np.diff(np.sort(all_z_dlas, axis=0), axis=0)
                    < self.min_z_separation,
                    axis=0,
                )
                self.sample_log_likelihoods[ind, num_lls, 0, 0] = np.nan

            # to prevent numerical underflow
            max_log_likelihood = np.nanmax(
                self.sample_log_likelihoods[:, num_lls, 0, 0]
            )

            sample_probabilities = np.exp(
                self.sample_log_likelihoods[:, num_lls, 0, 0] - max_log_likelihood
            )

            self.log_likelihoods[num_lls, 0, 0] = (
                max_log_likelihood
                + np.log(np.nanmean(sample_probabilities))
                - np.log(self.params.num_dla_samples) * num_lls
            )  # occams razor for more DLA parameters

            # if p(D | z_QSO, k DLA) is NaN, then
            # finish the loop.
            # It's usually because p(D | z_QSO, no DLA) is very high, so
            # the higher order DLA model likelihoods already underflowed
            if np.isnan(self.log_likelihoods[num_lls, 0, 0]):
                print(
                    "Finish the loop earlier because NaN value in log p(D | z_QSO, {} LLSs)".format(
                        num_lls
                    )
                )
                break

            # avoid nan values in the randsample weights
            nanind = np.isnan(sample_probabilities)
            W = sample_probabilities
            W[nanind] = 0.0

            self.base_sample_inds_lls[(num_lls - 1), :] = np.random.choice(
                np.arange(self.params.num_dla_samples).astype(np.int32),
                size=self.params.num_dla_samples,
                replace=True,
                p=W / W.sum(),
            )

        return self.log_likelihoods[:, 0, 0]

    def log_model_evidence_mgii(self, max_mgiis: int) -> np.ndarray:
        """
        Compute the log model evidence for the presence of MgII absorbers in the quasar spectra.
        """
        for num_mgii in range(1, max_mgiis + 1):
            # [Need to be parallelized]
            # Roman's code has this part to be parallelized.

            for i in range(self.params.num_dla_samples):
                # query the 1st parameter {z, logN}_{i=1}
                z_mgii = np.array([self.sample_z_mgiis[i]])
                log_nmgii = np.array([self.mgii_samples.log_nmgii_samples[i]])
                nmgii = np.array([self.mgii_samples.nmgii_samples[i]])

                # query the 2:k MgII parameters {z, logN}_{i=2}^k
                if num_mgii > 1:
                    base_ind = self.base_sample_inds_mgii[: (num_mgii - 1), i]

                    z_mgii_2_k = self.sample_z_mgiis[base_ind]
                    log_nmgii_2_k = self.mgii_samples.log_nmgii_samples[base_ind]
                    nmgii_2_k = self.mgii_samples.nmgii_samples[base_ind]

                    # append to samples to be applied on calculating the log likelihood
                    z_mgii = np.append(z_mgii, z_mgii_2_k)
                    log_nmgii = np.append(log_nmgii, log_nmgii_2_k)
                    nmgii = np.append(nmgii, nmgii_2_k)

                    del z_mgii_2_k, log_nmgii_2_k, nmgii_2_k

                # store the sample log likelihoods conditioned on k-MgIIs solely
                self.sample_log_likelihoods[
                    i, 0, num_mgii, 0
                ] = self.sample_log_likelihood_k_llss(
                    np.array([]),
                    np.array([]),
                    z_mgii,
                    nmgii,
                    np.array([]),
                    np.array([]),
                ) - np.log(
                    self.params.num_dla_samples
                )

            # check if any pair of absorbers in this sample is too close this has to
            # happen outside the parfor because "continue" slows things down
            # dramatically
            if num_mgii > 1:
                # all_z_mgii : (num_mgii, num_mgii_samples)
                ind = self.base_sample_inds_mgii[: (num_mgii - 1), :]

                all_z_mgiis = np.concatenate(
                    [self.sample_z_mgiis[None, :], self.sample_z_mgiis[ind]], axis=0
                )

                ind = np.any(
                    np.diff(np.sort(all_z_mgiis, axis=0), axis=0)
                    < self.min_z_separation,
                    axis=0,
                )
                self.sample_log_likelihoods[ind, 0, num_mgii, 0] = np.nan

            # to prevent numerical underflow
            max_log_likelihood = np.nanmax(
                self.sample_log_likelihoods[:, 0, num_mgii, 0]
            )

            sample_probabilities = np.exp(
                self.sample_log_likelihoods[:, 0, num_mgii, 0] - max_log_likelihood
            )

            self.log_likelihoods[0, num_mgii, 0] = (
                max_log_likelihood
                + np.log(np.nanmean(sample_probabilities))
                - np.log(self.params.num_dla_samples) * num_mgii
            )  # occams razor

            # if p(D | z_QSO, k MgII) is NaN, then
            # finish the loop.
            # It's usually because p(D | z_QSO, no MgII) is very high, so
            # the higher order MgII model likelihoods already underflowed
            if np.isnan(self.log_likelihoods[0, num_mgii, 0]):
                print(
                    "Finish the loop earlier because NaN value in log p(D | z_QSO, {} MgIIs)".format(
                        num_mgii
                    )
                )
                break

            # avoid nan values in the randsample weights
            nanind = np.isnan(sample_probabilities)
            W = sample_probabilities
            W[nanind] = 0.0

            self.base_sample_inds_mgii[(num_mgii - 1), :] = np.random.choice(
                np.arange(self.params.num_dla_samples).astype(np.int32),
                size=self.params.num_dla_samples,
                replace=True,
                p=W / W.sum(),
            )

        return self.log_likelihoods[0, :, 0]

    def log_model_evidence_civ(self, max_civs: int) -> np.ndarray:
        """
        Compute the log model evidence for the presence of CIV absorbers in the quasar spectra.
        """
        for num_civ in range(1, max_civs + 1):
            # [Need to be parallelized]
            # Roman's code has this part to be parallelized.
            # TODO: threading this part
            for i in range(self.params.num_dla_samples):
                # query the 1st parameter {z, logN}_{i=1}
                z_civ = np.array([self.sample_z_civs[i]])
                log_nciv = np.array([self.civ_samples.log_nciv_samples[i]])
                nciv = np.array([self.civ_samples.nciv_samples[i]])

                # query the 2:k CIV parameters {z, logN}_{i=2}^k
                if num_civ > 1:
                    base_ind = self.base_sample_inds_civ[: (num_civ - 1), i]

                    z_civ_2_k = self.sample_z_civs[base_ind]
                    log_nciv_2_k = self.civ_samples.log_nciv_samples[base_ind]
                    nciv_2_k = self.civ_samples.nciv_samples[base_ind]

                    # append to samples to be applied on calculating the log likelihood
                    z_civ = np.append(z_civ, z_civ_2_k)
                    log_nciv = np.append(log_nciv, log_nciv_2_k)
                    nciv = np.append(nciv, nciv_2_k)

                    del z_civ_2_k, log_nciv_2_k, nciv_2_k

                # store the sample log likelihoods conditioned on k-CIVs solely
                self.sample_log_likelihoods[
                    i, 0, 0, num_civ
                ] = self.sample_log_likelihood_k_llss(
                    np.array([]), np.array([]), np.array([]), np.array([]), z_civ, nciv
                ) - np.log(
                    self.params.num_dla_samples
                )  # additional occams razor

            # check if any pair of absorbers in this sample is too close this has to
            # happen outside the parfor because "continue" slows things down
            # dramatically
            if num_civ > 1:
                # all_z_civ : (num_civ, num_civ_samples)
                ind = self.base_sample_inds_civ[
                    : (num_civ - 1), :
                ]  # (num_civ - 1, num_civ_samples)

                all_z_civs = np.concatenate(
                    [self.sample_z_civs[None, :], self.sample_z_civs[ind]], axis=0
                )  # (num_civ, num_civ_samples)

                ind = np.any(
                    np.diff(np.sort(all_z_civs, axis=0), axis=0)
                    < self.min_z_separation,
                    axis=0,
                )
                self.sample_log_likelihoods[ind, 0, 0, num_civ] = np.nan

            # to prevent numerical underflow
            max_log_likelihood = np.nanmax(
                self.sample_log_likelihoods[:, 0, 0, num_civ]
            )

            sample_probabilities = np.exp(
                self.sample_log_likelihoods[:, 0, 0, num_civ] - max_log_likelihood
            )

            self.log_likelihoods[0, 0, num_civ] = (
                max_log_likelihood
                + np.log(np.nanmean(sample_probabilities))
                - np.log(self.params.num_dla_samples) * num_civ
            )  # occams razor

            # if p(D | z_QSO, k CIV) is NaN, then
            # finish the loop.
            # It's usually because p(D | z_QSO, no CIV) is very high, so
            # the higher order CIV model likelihoods already underflowed
            if np.isnan(self.log_likelihoods[num_civ, 0, 0]):
                print(
                    "Finish the loop earlier because NaN value in log p(D | z_QSO, {} CIVs)".format(
                        num_civ
                    )
                )
                break

            # avoid nan values in the randsample weights
            nanind = np.isnan(sample_probabilities)
            W = sample_probabilities
            W[nanind] = 0.0

            self.base_sample_inds_civ[(num_civ - 1), :] = np.random.choice(
                np.arange(self.params.num_dla_samples).astype(np.int32),
                size=self.params.num_dla_samples,
                replace=True,
                p=W / W.sum(),
            )

        return self.log_likelihoods[0, 0, :]

    def log_model_evidence_coupling(
        self, num_lls: int, num_mgii: int, num_civ: int
    ) -> np.ndarray:
        """
        Compute the log model evidence for the presence of LLSs, MgII, and CIV absorbers in the quasar spectra,
        at least two absorbers are in the model.

        Parameters:
        - num_lls: The number of LLS absorbers in the model.
        - num_mgii: The number of MgII absorbers in the model.
        - num_civ: The number of CIV absorbers in the model.
        """
        # [Need to be parallelized]
        # Roman's code has this part to be parallelized.

        # The assumption here is always search LLS first,
        # while the metals are the background absorbers as regularization.
        # TODO: Have a Laplace approximation for the posterior of the previous run.
        assert num_lls > 0
        for i in range(self.params.num_dla_samples):
            # LLS:
            # Non-informative prior (num_lls = 1)
            # + posterior from the previous run (num_lls > 1)
            # θ ~ p(θlls1)
            z_lls = np.array([self.sample_z_dlas[i]])
            log_nhis = np.array([self.dla_samples.log_nhi_samples[i]])
            nhis = np.array([self.dla_samples.nhi_samples[i]])
            # θ ~ p(θlls2 | D, θlls1)
            if num_lls > 1:
                # Here the definition of num_lls is different- it's the number of LLSs in the model
                base_ind_lls = self.base_sample_inds_lls[: (num_lls - 1), i]

                z_lls_2_k = self.sample_z_dlas[base_ind_lls]
                log_nhis_2_k = self.dla_samples.log_nhi_samples[base_ind_lls]
                nhis_2_k = self.dla_samples.nhi_samples[base_ind_lls]

                # append to samples to be applied on calculating the log likelihood
                z_lls = np.append(z_lls, z_lls_2_k)
                log_nhis = np.append(log_nhis, log_nhis_2_k)
                nhis = np.append(nhis, nhis_2_k)

                del z_lls_2_k, log_nhis_2_k, nhis_2_k

            # Metals: Here we use the posterior from the previous run to sample the parameters
            # θ ~ p(θmetal2 | D, θmetal1)

            # query the MgII parameter {z, logN}_{i=1}
            z_mgii = np.array([])
            log_nmgii = np.array([])
            nmgii = np.array([])
            # num = 1, then query p(θmetal2 | D, θmetal1)
            if num_mgii > 0:
                base_ind_mgii = self.base_sample_inds_mgii[: (num_mgii - 1), i]

                z_mgii_2_k = self.sample_z_mgiis[base_ind_mgii]
                log_nmgii_2_k = self.mgii_samples.log_nmgii_samples[base_ind_mgii]
                nmgii_2_k = self.mgii_samples.nmgii_samples[base_ind_mgii]

                # append to samples to be applied on calculating the log likelihood
                z_mgii = np.append(z_mgii, z_mgii_2_k)
                log_nmgii = np.append(log_nmgii, log_nmgii_2_k)
                nmgii = np.append(nmgii, nmgii_2_k)

                del z_mgii_2_k, log_nmgii_2_k, nmgii_2_k

            # query the CIV parameter {z, logN}_{i=1}
            z_civ = np.array([])
            log_nciv = np.array([])
            nciv = np.array([])
            if num_civ > 0:
                base_ind_civ = self.base_sample_inds_civ[: (num_civ - 1), i]

                z_civ_2_k = self.sample_z_civs[base_ind_civ]
                log_nciv_2_k = self.civ_samples.log_nciv_samples[base_ind_civ]
                nciv_2_k = self.civ_samples.nciv_samples[base_ind_civ]

                # append to samples to be applied on calculating the log likelihood
                z_civ = np.append(z_civ, z_civ_2_k)
                log_nciv = np.append(log_nciv, log_nciv_2_k)
                nciv = np.append(nciv, nciv_2_k)

                del z_civ_2_k, log_nciv_2_k, nciv_2_k

            # store the sample log likelihoods conditioned on k-LLSs, l-MgIIs, m-CIVs
            self.sample_log_likelihoods[
                i, num_lls, num_mgii, num_civ
            ] = self.sample_log_likelihood_k_llss(
                z_lls, nhis, z_mgii, nmgii, z_civ, nciv
            ) - np.log(
                self.params.num_dla_samples
            )  # additional occams razor

        # check if any pair of absorbers in this sample is too close
        # Multiple LLS:
        all_zs = np.concatenate([self.sample_z_dlas[None, :]], axis=0)
        if num_lls > 1:
            ind = self.base_sample_inds_lls[: (num_lls - 1), :]
            all_zs = np.concatenate([all_zs, self.sample_z_dlas[ind]], axis=0)
        # Metals:
        if num_mgii > 0:
            ind = self.base_sample_inds_mgii[: (num_mgii - 1), :]
            all_z_mgiis = self.sample_z_mgiis[ind]  # (num_mgii, num_mgii_samples)
            # convert mgii redshifts to hi redshifts
            all_z_mgiis = (
                all_z_mgiis + 1
            ) * self.params.mgii_2796_wavelength / self.params.lya_wavelength - 1
            all_zs = np.concatenate([all_zs, all_z_mgiis], axis=0)
        if num_civ > 0:
            ind = self.base_sample_inds_civ[: (num_civ - 1), :]
            all_z_civs = self.sample_z_civs[ind]  # (num_civ, num_civ_samples)
            # convert civ redshifts to hi redshifts
            all_z_civs = (
                all_z_civs + 1
            ) * self.params.civ_1548_wavelength / self.params.lya_wavelength - 1
            all_zs = np.concatenate([all_zs, all_z_civs], axis=0)

        # Make NaNs to those samples that are too close
        ind = np.any(
            np.diff(np.sort(all_zs, axis=0), axis=0) < self.min_z_separation, axis=0
        )
        self.sample_log_likelihoods[ind, num_lls, num_mgii, num_civ] = np.nan

        # to prevent numerical underflow
        max_log_likelihood = np.nanmax(
            self.sample_log_likelihoods[:, num_lls, num_mgii, num_civ]
        )

        sample_probabilities = np.exp(
            self.sample_log_likelihoods[:, num_lls, num_mgii, num_civ]
            - max_log_likelihood
        )

        self.log_likelihoods[num_lls, num_mgii, num_civ] = (
            max_log_likelihood
            + np.log(np.nanmean(sample_probabilities))
            - np.log(self.params.num_dla_samples) * (num_lls + num_mgii + num_civ)
        )  # occams razor

        # TODO: resample the base inds for the next run (if necessary)

        return self.log_likelihoods[num_lls, num_mgii, num_civ]

    def log_model_evidences(
        self,
        max_lls: int = 2,
        max_mgiis: int = 2,
        max_civs: int = 2,
    ) -> np.ndarray:
        """
        Compute the log model evidences for the presence of LLSs in the quasar spectra,
        considering the alternative metal absorption lines (MgII, CIV).

        Current parameter prior consideration for multiple absorbers:
        p(θ1, θ2 | D) = p(θ1 | D) * p(θ2 | D, θ1) ≈ p(θ1) * p(θ2 | D, θ1)

        For a combination of LLSs and MgII/CIV absorbers:
        p(θlls, θmetal | D) = p(θlls | D) * p(θmetal | D, θlls) # the dependence here is for not overlapping absorbers
            = p(θlls | D) * p(θmetal | D)
            ≈ p(θlls) * p(θmetal2 | D, θmetal1)

        p(θmetal2 | D, θmetal1) is the posterior from the previous run.

        For a combination of LLSs, MgII, and CIV absorbers:
        p(θlls, θmgii, θciv | D) = p(θlls | D) * p(θmgii | D, θlls) * p(θciv | D, θlls, θmgii)
            = p(θlls | D) * p(θmgii | D) * p(θciv | D)
            ≈ p(θlls) * p(θmgii2 | D, θmgii1) * p(θciv2 | D, θciv1)

        Parameters:
        - max_lls: The maximum number of LLSs to consider in the model.
        - max_mgiis: The maximum number of MgII absorbers to consider in the model.
        - max_civs: The maximum number of CIV absorbers to consider in the model.

        Returns:
        - log_likelihoods: The log model evidences for the presence of LLSs in the quasar spectra.
        """
        # allocate the final log model evidences.
        # log p(D | z_QSO, k LLSs, k MgII, k CIV),
        # where k = 0, 1, ..., max_k
        self.log_likelihoods = np.empty((max_lls + 1, max_mgiis + 1, max_civs + 1))
        self.log_likelihoods[:] = np.nan

        # base inds to store the QMC samples to be resampled according
        # the prior, which is the posterior of the previous run.
        self.base_sample_inds_lls = np.zeros(
            (
                max_lls,
                self.params.num_dla_samples,
            ),
            dtype=np.int32,
        )
        self.base_sample_inds_mgii = np.zeros(
            (
                max_mgiis,
                self.params.num_dla_samples,
            ),
            dtype=np.int32,
        )
        self.base_sample_inds_civ = np.zeros(
            (
                max_civs,
                self.params.num_dla_samples,
            ),
            dtype=np.int32,
        )

        # sorry, let me follow the convention of the MATLAB code here
        # could be changed to (max_dlas, num_dla_samples) in the future.
        self.sample_log_likelihoods = np.empty(
            (self.params.num_dla_samples, max_lls + 1, max_mgiis + 1, max_civs + 1)
        )
        self.sample_log_likelihoods[:] = np.nan

        # prepare z_dla samples
        self.sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )
        self.sample_z_civs = self.civ_samples.sample_z_civs(
            self.this_wavelengths, self.z_qso
        )
        self.sample_z_mgiis = self.mgii_samples.sample_z_mgiis(
            self.this_wavelengths, self.z_qso
        )

        # compute the null model evidence
        log_likelihood_null = self.log_model_evidence()
        self.log_likelihoods[0, 0, 0] = log_likelihood_null
        print("Null model evidence: {}".format(log_likelihood_null))
        # compute probabilities under single absorber models (LLS, CIV, MgII)
        # for each of the sampled (normalized offset, log(N HI)) pairs
        print("Computing log model evidence for LLSs")
        log_likelihood_lls = self.log_model_evidence_lls(max_lls)
        print("LLS model evidence: {}".format(log_likelihood_lls))

        print("Computing log model evidence for MgII")
        log_likelihood_mgii = self.log_model_evidence_mgii(max_mgiis)
        print("MgII model evidence: {}".format(log_likelihood_mgii))

        print("Computing log model evidence for CIV")
        log_likelihood_civ = self.log_model_evidence_civ(max_civs)
        print("CIV model evidence: {}".format(log_likelihood_civ))

        # compute the log model evidence for coupling terms:
        # Generate combinations and filter out the undesired ones
        combinations = [
            (i, j, k)
            for i, j, k in itertools.product(
                range(0, max_lls + 1), range(0, max_mgiis + 1), range(0, max_civs + 1)
            )
            if not ((i == 0 and j == 0) or (i == 0 and k == 0) or (j == 0 and k == 0))
        ]

        for combo in combinations:
            # lls, mgii, civ
            i, j, k = combo
            # skip the no LLS case
            if i == 0:
                print("Skip the case of no LLS")
                continue

            # compute the log model evidence for the coupling terms
            print(
                "Computing log model evidence for LLSs: {}, MgII: {}, CIV: {}".format(
                    i, j, k
                )
            )
            self.log_model_evidence_coupling(i, j, k)
            print("... {}".format(self.log_likelihoods[i, j, k]))

        return self.log_likelihoods

    def log_priors_mgii(self, z_qso: float, max_mgii: int) -> np.ndarray:
        """
        Compute the log priors for MgII absorbers in the quasar spectra.
        """
        log_priors = self.log_priors(z_qso=z_qso, max_dlas=max_mgii)
        # Approximate
        # TODO: do the exact number
        log_priors += 7 / np.log10(np.exp(1))

        return log_priors

    def log_priors_civ(self, z_qso: float, max_civ: int) -> np.ndarray:
        """
        Compute the log priors for CIV absorbers in the quasar spectra.
        """
        log_priors = self.log_priors(z_qso=z_qso, max_dlas=max_civ)
        # Approximate
        # TODO: do the exact number
        log_priors += 5 / np.log10(np.exp(1))

        return log_priors

    # A function to get the maximum posterior model and the corresponding
    # maximum a posteriori (MAP) parameters
    def maximum_a_posteriori(
        self,
        log_posteriors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the maximum a posteriori (MAP) parameters for the maximum posterior model

        Parameters
        ----------
        log_posteriors : np.ndarray
            The log posteriors of the models

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The maximum a posteriori (MAP) parameters for the maximum posterior model
        """
        # Get the indices of the maximum log posterior
        i, j, k = np.unravel_index(np.nanargmax(log_posteriors), log_posteriors.shape)

        print("Maximum posterior model: LLS={}, MgII={}, CIV={}".format(i, j, k))

        # Get the maximum posteriori
        maxind = np.nanargmax(self.sample_log_likelihoods[:, i, j, k])

        # MgII
        MAP_log_nmgii = np.array([])
        MAP_z_mgiis = np.array([])
        if j > 0:
            ind = self.base_sample_inds_mgii[:j, maxind]
            MAP_log_nmgii = np.append(
                MAP_log_nmgii, self.mgii_samples.log_nmgii_samples[ind]
            )
            MAP_z_mgiis = np.append(MAP_z_mgiis, self.sample_z_mgiis[ind])
        # CIV
        MAP_log_nciv = np.array([])
        MAP_z_civ = np.array([])
        if k > 0:
            ind = self.base_sample_inds_civ[:k, maxind]
            MAP_log_nciv = np.append(
                MAP_log_nciv, self.civ_samples.log_nciv_samples[ind]
            )
            MAP_z_civ = np.append(MAP_z_civ, self.sample_z_civs[ind])

        # LLS
        MAP_log_nhi = self.dla_samples.log_nhi_samples[maxind]
        MAP_z_lya = self.sample_z_dlas[maxind]

        for _i in range(i - 1):
            ind = self.base_sample_inds_lls[_i, maxind]
            MAP_log_nhi = np.append(MAP_log_nhi, self.dla_samples.log_nhi_samples[ind])
            MAP_z_lya = np.append(MAP_z_lya, self.sample_z_dlas[ind])

        return (
            MAP_log_nhi,
            MAP_z_lya,
            MAP_log_nmgii,
            MAP_z_mgiis,
            MAP_log_nciv,
            MAP_z_civ,
        )
