"""
A GP class for having at subDLAs intervening in a given slightline.

This is basically the same as .dla_gp, but has different priors on
logNHI and model prior. In order to not overwrite the class method
in .dla_gp.DLAGP, I chose to inherent from .null_gp.NullGP.
"""

from typing import Tuple, Optional
import numpy as np
import h5py

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .dla_gp import DLAGP
from .voigt import voigt_absorption
from .subdla_samples import SubDLASamplesMAT  # for convenient autocomplete


class SubDLAGP(DLAGP):
    """
    SubDLA GP model for QSO emission + subDLA intervening:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, z_dla, logNHI)

    additional two parameters (z_dla, logNHI) will control the position
    and the strength of the absorption intervening on the QSO emission.

    SubDLA parameter prior: logNHI ~ U(19.5, 20)

    Since the integration is not tractable, we use Quasi Monte Carlo (QMC) to approximate
    the model evidence.

    The number of QMC samples is defined in Parameters and DLASamples.

    :param rest_wavelengths: λ, the range of λ you model your GP on QSO emission
    :param mu: mu, the mean model of the GP.
    :param M: M, the low-rank decomposition of the covariance kernel: K = MM^T.
    :param log_omega: log ω, the pixel-wise noise of the model. Used to model absorption noise.
    :param log_c_0: log c₀, the constant in the Lyman forest noise model.
    :param log_tau_0: log τ₀, the scale factor of effective optical depth in the absorption noise.
    :param log_beta: log β, the exponent of the effective optical depth in the absorption noise.
    :param prev_tau_0: τ_kim, the scale factor of effective optical depth used in mean-flux suppression.
    :param prev_beta: β_kim, the exponent of the effective optical depth used in mean-flux suppression.

    Future: MCMC can be embedded in the class as an instance method.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: SubDLASamplesMAT,
        rest_wavelengths: np.ndarray,
        mu: np.ndarray,
        M: np.ndarray,
        log_omega: np.ndarray,
        log_c_0: float,
        log_tau_0: float,
        log_beta: float,
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
        min_z_separation: float = 3000.0,
        broadening: bool = True,
    ):
        # Initialize the DLAGP class with explicit argument passing
        super().__init__(
            params=params,
            prior=prior,
            dla_samples=dla_samples,
            rest_wavelengths=rest_wavelengths,
            mu=mu,
            M=M,
            log_omega=log_omega,
            log_c_0=log_c_0,
            log_tau_0=log_tau_0,
            log_beta=log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )

    def log_priors(self, z_qso: float, max_dlas: int) -> float:
        """
        Get the model prior for the SubDLA model, defined as:
            P(k subDLA | zQSO) = P(at least k subDLAs | zQSO) - P(at least (k + 1) subDLAs | zQSO)

        Where:
            P(at least 1 subDLA | zQSO) = Z_lls / Z_dla * M / N

        Here:
        - M is the number of subDLAs below this zQSO.
        - N is the number of quasars below this zQSO.
        - Z_lls and Z_dla are normalization factors for subDLAs and DLAs.

        Args:
            z_qso (float): The redshift of the quasar.
            max_dlas (int): The maximum number of subDLAs considered.

        Returns:
            log_priors_dla (float): The log prior for each subDLA.
        """
        this_num_dlas, this_num_quasars = self.prior.less_ind(z_qso)

        # Adjust the prior for subDLAs using the Z_lls / Z_dla ratio
        p_dlas = (
            self.dla_samples._Z_lls
            / self.dla_samples._Z_dla
            * (this_num_dlas / this_num_quasars) ** np.arange(1, max_dlas + 1)
        )

        # Adjust the probabilities to account for P(k subDLA | zQSO)
        for i in range(max_dlas - 1):
            p_dlas[i] = p_dlas[i] - p_dlas[i + 1]

        log_priors_dla = np.log(p_dlas)

        return log_priors_dla


class SubDLAGPMAT(SubDLAGP):
    """
    Load a learned model from a .mat file for SubDLA GP.

    The learned model file structure is the same as DLAGP.
    The sample file differs for subDLAs.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: SubDLASamplesMAT,
        min_z_separation: float = 3000.0,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        broadening: bool = True,
    ):
        # Load the learned model from the .mat file
        with h5py.File(learned_file, "r") as learned:
            rest_wavelengths = learned["rest_wavelengths"][:, 0]
            mu = learned["mu"][:, 0]
            M = learned["M"][()].T
            log_omega = learned["log_omega"][:, 0]
            log_c_0 = learned["log_c_0"][0, 0]
            log_tau_0 = learned["log_tau_0"][0, 0]
            log_beta = learned["log_beta"][0, 0]

        # Initialize the SubDLAGP class explicitly with all parameters
        super().__init__(
            params=params,
            prior=prior,
            dla_samples=dla_samples,
            rest_wavelengths=rest_wavelengths,
            mu=mu,
            M=M,
            log_omega=log_omega,
            log_c_0=log_c_0,
            log_tau_0=log_tau_0,
            log_beta=log_beta,
            prev_tau_0=0.0023,
            prev_beta=3.65,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )
