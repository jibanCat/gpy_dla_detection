"""
A class to handle the Null model in Ho, Bird, Garnett (2020).
"""
import numpy as np
import h5py
import scipy
from scipy import interpolate
from scipy.linalg import lapack

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .effective_optical_depth import effective_optical_depth


class NullGP:
    """
    Null GP model for QSO emission:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim)
    
    :param rest_wavelengths: λ, the range of λ you model your GP on QSO emission
    :param mu: mu, the mean model of the GP.
    :param M: M, the low rank decomposition of the covariance kernel: K = MM^T.
    :param log_omega: log ω, the pixel-wise noise of the model. Used to model absorption noise.
    :param log_c_0: log c₀, the constant in the Lyman forest noise model,
        Lyman forest noise := s(z) = 1 - exp(-effective_optical_depth) + c_0.
    :param log_tau_0: log τ₀, the scale factor of effective optical depth in the absorption noise,
        effective_optical_depth := ∑ τ₀ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
    :param log_beta: log β, the exponent of the effective optical depth in the absorption noise.
    :param prev_tau_0: τ_kim, the scale factor of effective optical depth used in mean-flux suppression.
    :param prev_beta: β_kim, the exponent of the effective optical depth used in mean-flux suppression. 
    
    Note: we assume we load the learned GP from a .mat file, but this could be deprecate in the future  
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        rest_wavelengths: np.ndarray,
        mu: np.ndarray,
        M: np.ndarray,
        log_omega: np.ndarray,
        log_c_0: float,
        log_tau_0: float,
        log_beta: float,
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
    ):
        # parameters and model priors
        self.params = params
        self.prior = prior

        # learned GP parameters
        self.rest_wavelengths = rest_wavelengths
        self.mu = mu
        self.M = M
        self.log_omega = log_omega
        self.log_c_0 = log_c_0
        self.log_tau_0 = log_tau_0
        self.log_beta = log_beta

        # mean-flux suppression
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta

        # preprocess model interpolants
        self.mu_interpolator = interpolate.interp1d(rest_wavelengths, mu)
        self.log_omega_interpolator = interpolate.interp1d(rest_wavelengths, log_omega)
        self.list_M_interpolators = [
            interpolate.interp1d(rest_wavelengths, eigenvector) for eigenvector in M.T
        ]

    def M_interpolator(self, this_rest_wavelengths: np.ndarray) -> np.ndarray:
        """
        The interpolant for low-rank decomposition of the covariance matrix.

        M_interpolator = interpolate
            griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');

        TODO: this needs to be validated. The original MATLAB format has some sort
        of unspoken wisdom. I assume it is doing 1-D interpolation on each eigenvector.

        :param this_rest_wavelengths: (n_points, ) the rest wavelengths of the observed data.
        :return M: (n_points, k) the interpolated low-rank decomposition of the covariance matrix.
        """
        assert len(self.list_M_interpolators) == self.params.k

        M_T = np.empty((self.params.k, this_rest_wavelengths.shape[0]))

        for i, m_interpolator in enumerate(self.list_M_interpolators):
            M_T[i, :] = m_interpolator(this_rest_wavelengths)

        return M_T.T

    def set_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        noise_variance: np.ndarray,
        pixel_mask: np.ndarray,
        z_qso: float,
        normalize: bool = True,
        build_model: bool = True,
    ) -> None:
        """
        Set "testing" data to be evaluated. Now assumed to be a single
        spectrum, but we can implement a batch of spectra in the future.

        :param X: (n_points, ) this_rest_wavelengths, the emission wavelengths of the spectrum.
        :param Y: (n_points, ) this_flux, the flux of the observed spectrum
        :param noise_variance: (n_points, ) the instrumental noise variance per pixel.
        :param pixel_mask: (n_points, ) the pixel mask corresponding to the read_spec you used.
        :param z_qso: the redshift of the quasar.

        Note: n_points represents number of pixels within a spectrum
        """
        self.x = X
        self.y = Y
        self.v = noise_variance
        self.pixel_mask = pixel_mask
        self.z_qso = z_qso

        # normalize flux: be aware that you might choose a normalization range
        # out side of the spectrum range.
        if normalize:
            ind = (
                (self.x >= self.params.normalization_min_lambda)
                & (self.x <= self.params.normalization_max_lambda)
                & (~pixel_mask)
            )
            this_median = np.nanmedian(self.y[ind])
            self.y = self.y / this_median
            self.v = self.v / this_median ** 2

        # apply pixel mask and filter spectrum within modelling range
        ind = (self.x >= self.params.min_lambda) & (self.x <= self.params.max_lambda)
        self.ind_unmasked = ind

        # keep complete copy of equally spaced wavelengths for absorption
        # computation; supposed to be observed wavelengths
        self.unmasked_wavelengths = Parameters.observed_wavelengths(self.x, z_qso)[ind]
        # filter pixel mask
        ind = ind & (~self.pixel_mask)
        self.this_wavelengths = Parameters.observed_wavelengths(self.x, z_qso)[ind]
        self.x = self.x[ind]
        self.y = self.y[ind]
        self.v = self.v[ind]

        self.ind = ind

        if build_model:
            self.get_interp(self.x, self.y, self.this_wavelengths, self.z_qso)

        # ensure enough pixels are on either side for convolving with
        # instrument profile
        self.padded_wavelengths = np.concatenate(
            [
                np.logspace(
                    np.log10(self.unmasked_wavelengths.min())
                    - self.params.width * self.params.pixel_spacing,
                    np.log10(self.unmasked_wavelengths.min())
                    - self.params.pixel_spacing,
                    self.params.width,
                ),
                self.unmasked_wavelengths,
                np.logspace(
                    np.log10(self.unmasked_wavelengths.max())
                    + self.params.pixel_spacing,
                    np.log10(np.max(self.unmasked_wavelengths))
                    + self.params.width * self.params.pixel_spacing,
                    self.params.width,
                ),
            ]
        )

    def get_interp(
        self, x: np.ndarray, y: np.ndarray, wavelengths: np.ndarray, z_qso: float
    ) -> None:
        """
        Build and interpolate the GP model onto the observed data.
        
        p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
        
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

        this_log_omega = self.log_omega_interpolator(x)
        this_omega2 = np.exp(2 * this_log_omega)

        # set Lyseries absorber redshift for mean-flux suppression
        # apply the lya_absorption after the interpolation because NaN will appear in this_mu
        total_optical_depth = effective_optical_depth(
            wavelengths,
            self.prev_beta,
            self.prev_tau_0,
            z_qso,
            self.params.num_forest_lines,
        )
        # total absorption effect of Lyseries absorption on the mean-flux
        lya_absorption = np.exp(-np.sum(total_optical_depth, axis=1))

        this_mu = this_mu * lya_absorption
        this_M = this_M * lya_absorption[:, None]
        assert this_M.shape[1] == self.params.k

        # set another Lyseries absorber redshift to use in covariance
        lya_optical_depth = effective_optical_depth(
            wavelengths,
            np.exp(self.log_beta),
            np.exp(self.log_tau_0),
            z_qso,
            self.params.num_forest_lines,
        )

        this_scaling_factor = (
            1 - np.exp(-np.sum(lya_optical_depth, axis=1)) + np.exp(self.log_c_0)
        )

        # this is the omega included the Lyseries
        this_omega2 = this_omega2 * this_scaling_factor ** 2

        # re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
        # now the null model likelihood is:
        # p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
        this_omega2 = this_omega2 * lya_absorption ** 2

        # assign to instance attrs
        self.this_mu = this_mu
        self.this_M = this_M
        self.this_omega2 = this_omega2

    @property
    def mean(self):
        """
        mean model not yet interpolated onto data
        """
        return self.mu

    @property
    def K(self):
        """
        covariance kernel not yet interpolated onto data
        """
        return np.matmul(self.M, self.M.T)

    @property
    def this_mean(self):
        """
        mean model interpolated onto data
        """
        return self.this_mu

    @property
    def this_noise(self):
        """
        noise kernel: absorption noise (omega2) + instrumental noise
        """
        return self.this_omega2 + self.v

    @property
    def this_K(self):
        """
        covariance kernel: K = M M'
        """
        return np.matmul(self.this_M, self.this_M.T)

    @property
    def X(self) -> np.ndarray:
        return self.x

    @property
    def Y(self) -> np.ndarray:
        return self.y

    @property
    def V(self) -> np.ndarray:
        """
        Noise variance matrix, only diagonal elements.
        """
        return self.v

    def log_model_evidence(self) -> float:
        """
        Compute the log model evidence with the learned model.
        
        Note: assume the model has already interpolated onto the observed data,
        and data got loaded into the GP instance.        
        """
        log_likelihood_no_dla = self.log_mvnpdf_low_rank(
            self.y, self.this_mu, self.this_M, self.this_omega2 + self.v
        )

        return log_likelihood_no_dla

    @staticmethod
    def log_mvnpdf_low_rank(
        y: np.ndarray, mu: np.ndarray, M: np.ndarray, d: np.ndarray, scipy_lapack: bool = True
    ) -> float:
        """
        efficiently computes
        
           log N(y; mu, MM' + diag(d))
        
        :param y: this_flux, (n_points, )
        :param mu: this_mu, the mean vector of GP, (n_points, )
        :param M: this_M, the low rank decomposition of covariance matrix, (n_points, k)
        :param d: diagonal noise term, (n_points, )
        """
        log_2pi = 1.83787706640934534

        n, k = M.shape

        y = y[:, None] - mu[:, None]

        d_inv = 1 / d[:, None]  # (n_points, 1)
        D_inv_y = d_inv * y  # (n_points, 1)
        D_inv_M = d_inv * M  # (n_points, k)

        # use Woodbury identity, define
        #   B = (I + M' D^-1 M),
        # then
        #   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1
        B = np.matmul(M.T, D_inv_M)  # (k, n_points) * (n_points, k) -> (k, k)
        # add the identity matrix with magic indicing
        B.ravel()[0 :: (k + 1)] = B.ravel()[0 :: (k + 1)] + 1
        # numpy cholesky returns lower triangle, different than MATLAB's upper triangle
        L = np.linalg.cholesky(B)
        # C = B^-1 M' D^-1
        if scipy_lapack:
            tmp = np.matmul(lapack.dtrtri(np.asfortranarray(L), lower=1)[0], D_inv_M.T)
            C = np.matmul(lapack.dtrtri(np.asfortranarray(L.T), lower=0)[0], tmp)
        else:
            tmp = scipy.linalg.solve_triangular(
                L, D_inv_M.T, lower=True
            )  # (k, n_points)
            C = scipy.linalg.solve_triangular(L.T, tmp, lower=False)  # (k, n_points)

        K_inv_y = D_inv_y - np.matmul(D_inv_M, np.matmul(C, y))  # (n_points, 1)

        log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))

        log_p = -0.5 * (np.matmul(y.T, K_inv_y).sum() + log_det_K + n * log_2pi)

        return log_p

    def log_prior(self, z_qso: float, without_subDLAs: bool = True) -> float:
        """
        get the model prior of null model, this is defined to be:
            P(no DLA | zQSO) = 1 - P(DLA | zQSO) - P(subDLA | zQSO),
        
        where

            P(DLA | zQSO) = M / N
        
        M : number of DLAs below this zQSO
        N : number of quasars below this zQSO

        without P(subDLA | zQSO),
            P(DLA | zQSO) = 1 - M / N
        
        we return the null model prior without subDLAs prior here (
        as defined in the Garnett (2017) code), but remember to substract
        the subDLA prior if you want to take into account subDLA prior as
        in Ho, Bird, Garnett (2020). 
        """
        if not without_subDLAs:
            NotImplementedError

        this_num_dlas, this_num_quasars = self.prior.less_ind(z_qso)

        return np.log(1 - (this_num_dlas / this_num_quasars))


class NullGPMAT(NullGP):
    """
    Load learned model from .mat file
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
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
            prev_tau_0=0.0023,
            prev_beta=3.65,
        )
