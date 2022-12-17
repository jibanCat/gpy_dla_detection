"""
civ_set_parameters.py : sets various parameters for the CIV GP
pipeline.

See the original code here for mode details:
https://github.com/rezamonadi/GaussianProcessCIV

Note:
----
Physical constants are handled as class attrs
Pipeline parameters are handled as instance attrs
Lambda functions are handled as instance methods
"""
import numpy as np

from .set_parameters import Parameters


class CIVParameters(Parameters):


    civ_1548_wavelength = 1548.2040		 # CIV transition wavelength  Å
    civ_1550_wavelength = 1550.77810

    def __init__(
        self,
        # normalization parameters
        loading_min_lambda: float = 1310.0,  # range of rest wavelengths to load  Å
        loading_max_lambda: float = 1555.0,
        normalization_min_lambda: float = 1420,  # range of rest wavelengths to use   Å
        normalization_max_lambda: float = 1475,  # for flux normalization
        # null model parameters
        min_lambda: float = 1311.0,  # range of rest wavelengths to       Å
        max_lambda: float = 1554.0,  #   model
        dlambda: float = 0.5,  # separation of wavelength grid      Å
        k: int = 20,  # rank of non-diagonal contribution
        max_noise_variance: float = 4.0
        ** 2,  # maximum pixel noise allowed during model training
        num_civ_samples: int = 10000,  # number of parameter samples
        z_qso_cut: float = 1.7,
        min_num_pixels: int = 400,
        uniform_min_log_nciv: float = 12.88,
        uniform_max_log_nciv: float = 14.5,
        fit_min_log_nciv: float = 12.88,
        fit_max_log_nciv: float = 15,
        prior_z_qso_increase: float = 30000,
        pixel_spacing: float = 1e-4, 
        num_lines: int = 2,
        max_z_cut: float = 3000,
        min_z_cut: float = 3000,
        width: int = 3, # width of Gaussian broadening (# pixels)
        minFunc_options: dict = {  # optimization options for model fitting
            "MaxIter": 10000,
            "MaxFunEvals": 10000,
        },
    ):
        self.loading_min_lambda = loading_min_lambda
        self.loading_max_lambda = loading_max_lambda

        self.z_qso_cut = z_qso_cut
        self.min_num_pixels = min_num_pixels

        self.normalization_min_lambda = normalization_min_lambda
        self.normalization_max_lambda = normalization_max_lambda

        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.dlambda = dlambda
        self.k = k
        self.max_noise_variance = max_noise_variance

        self.minFunc_options = minFunc_options

        self.num_civ_samples = num_civ_samples
        # self.alpha = alpha
        self.uniform_min_log_nciv = uniform_min_log_nciv
        self.uniform_max_log_nciv = uniform_max_log_nciv
        self.fit_min_log_nciv = fit_min_log_nciv
        self.fit_max_log_nciv = fit_max_log_nciv

        self.prior_z_qso_increase = self.kms_to_z(prior_z_qso_increase)

        self.width = width
        self.pixel_spacing = pixel_spacing

        self.num_lines = num_lines
        self.max_z_cut = self.kms_to_z(max_z_cut)
        self.min_z_cut = self.kms_to_z(min_z_cut)

    def max_z_civ(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines maximum z_CIV to search

        We only consider z_CIV within the modelling range.
        """
        # rest_wavelengths = self.emitted_wavelengths(wavelengths, z_qso)
        # ind = (rest_wavelengths >= self.min_lambda) & (
        #     rest_wavelengths <= self.max_lambda
        # )
        return z_qso - self.max_z_cut

    def min_z_civ(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines minimum z_CIV to search

        We only consider z_CIV within the modelling range.
        """
        rest_wavelengths = self.emitted_wavelengths(wavelengths, z_qso)
        ind = (rest_wavelengths >= self.min_lambda) & (
            rest_wavelengths <= self.max_lambda
        )
        return np.max(
            [
                np.min(wavelengths[ind]) / self.civ_1548_wavelength - 1,
                self.observed_wavelengths(1310, z_qso) / self.civ_1548_wavelength - 1
                + self.min_z_cut,
            ]
        )
