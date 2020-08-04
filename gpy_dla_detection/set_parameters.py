"""
set_parameters.py : sets various parameters for the DLA detection
pipeline.

Note:
----
Physical constants are handled as class attrs
Pipeline parameters are handled as instance attrs
Lambda functions are handled as instance methods
"""
import numpy as np


class Parameters:
    # physical constants
    lya_wavelength: float = 1215.6701  # Lyman alpha transition wavelength  Å
    lyb_wavelength: float = 1025.7223  # Lyman beta  transition wavelength  Å
    lyman_limit: float = 911.7633  # Lyman limit wavelength             Å
    speed_of_light: float = 299792458.0  # speed of light                     m s⁻¹

    def __init__(
        self,
        # file loading parameters
        loading_min_lambda: float = 910.0,  # range of rest wavelengths to load  Å
        loading_max_lambda: float = 1217.0,
        # preprocessing parameters
        z_qso_cut: float = 2.15,  # filter out QSOs with z less than this threshold
        min_num_pixels: int = 200,  # minimum number of non-masked pixels
        # normalization parameters
        normalization_min_lambda: float = 1310.0,  # range of rest wavelengths to use   Å
        normalization_max_lambda: float = 1325.0,  #   for flux normalization
        # null model parameters
        min_lambda: float = 911.75,  # range of rest wavelengths to       Å
        max_lambda: float = 1215.75,  #   model
        dlambda: float = 0.25,  # separation of wavelength grid      Å
        k: int = 20,  # rank of non-diagonal contribution
        max_noise_variance: float = 3.0
        ** 2,  # maximum pixel noise allowed during model training
        # optimization parameters
        initial_c_0: float = 0.1,  # initial guess for c₀
        initial_tau_0: float = 0.0023,  # initial guess for τ₀
        initial_beta: float = 3.65,  # initial guess for β
        minFunc_options: dict = {  # optimization options for model fitting
            "MaxIter": 2000,
            "MaxFunEvals": 4000,
        },
        # DLA model parameters: parameter samples
        num_dla_samples: int = 10000,  # number of parameter samples
        alpha: float = 0.97,  # weight of KDE component in mixture
        uniform_min_log_nhi: float = 20.0,  # range of column density samples    [cm⁻²]
        uniform_max_log_nhi: float = 23.0,  # from uniform distribution
        fit_min_log_nhi: float = 20.0,  # range of column density samples    [cm⁻²]
        fit_max_log_nhi: float = 22.0,  # from fit to log PDF
        # model prior parameters
        prior_z_qso_increase: float = 30000.0,  # use QSOs with z < (z_QSO + x) for prior
        # instrumental broadening parameters
        width: int = 3,  # width of Gaussian broadening (# pixels)
        pixel_spacing: float = 1e-4,  # wavelength spacing of pixels in dex
        # DLA model parameters: absorber range and model
        num_lines: int = 3,  # number of members of the Lyman series to use
        max_z_cut: float = 3000.0,  # max z_DLA = z_QSO - max_z_cut
        min_z_cut: float = 3000.0,  # min z_DLA = z_Ly∞ + min_z_cut
        # Lyman-series array: for modelling the forests of Lyman series
        num_forest_lines: int = 31,
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

        self.initial_c_0 = initial_c_0
        self.initial_tau_0 = initial_tau_0
        self.initial_beta = initial_beta
        self.minFunc_options = minFunc_options

        self.num_dla_samples = num_dla_samples
        self.alpha = alpha
        self.uniform_min_log_nhi = uniform_min_log_nhi
        self.uniform_max_log_nhi = uniform_max_log_nhi
        self.fit_min_log_nhi = fit_min_log_nhi
        self.fit_max_log_nhi = fit_max_log_nhi

        self.prior_z_qso_increase = self.kms_to_z(prior_z_qso_increase)

        self.width = width
        self.pixel_spacing = pixel_spacing

        self.num_lines = num_lines
        self.max_z_cut = self.kms_to_z(max_z_cut)
        self.min_z_cut = self.kms_to_z(min_z_cut)

        self.num_forest_lines = num_forest_lines

    @classmethod
    def kms_to_z(cls, kms: float) -> float:
        """
        converts relative velocity in km s^-1 to redshift difference
        """
        return (kms * 1000) / cls.speed_of_light

    @staticmethod
    def emitted_wavelengths(observed_wavelengths: np.ndarray, z: float) -> np.ndarray:
        """
        utility functions for redshifting
        """
        return observed_wavelengths / (1 + z)

    @staticmethod
    def observed_wavelengths(emitted_wavelengths: np.ndarray, z: float) -> np.ndarray:
        """
        utility functions for redshifting
        """
        return emitted_wavelengths * (1 + z)

    def max_z_dla(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines maximum z_DLA to search

        We only consider z_dla within the modelling range.
        """
        rest_wavelengths = self.emitted_wavelengths(wavelengths, z_qso)
        ind = (rest_wavelengths >= self.min_lambda) & (
            rest_wavelengths <= self.max_lambda
        )
        return (np.max(wavelengths[ind]) / self.lya_wavelength - 1) - self.max_z_cut

    def min_z_dla(self, wavelengths: np.ndarray, z_qso: float) -> float:
        """
        determines minimum z_DLA to search

        We only consider z_dla within the modelling range.
        """
        rest_wavelengths = self.emitted_wavelengths(wavelengths, z_qso)
        ind = (rest_wavelengths >= self.min_lambda) & (
            rest_wavelengths <= self.max_lambda
        )
        return np.max(
            [
                np.min(wavelengths[ind]) / self.lya_wavelength - 1,
                self.observed_wavelengths(self.lyman_limit, z_qso) / self.lya_wavelength
                - 1
                + self.min_z_cut,
            ]
        )

    def __repr__(self):
        """
        print out the default pipeline parameters
        """
        return str(self.__dict__)
