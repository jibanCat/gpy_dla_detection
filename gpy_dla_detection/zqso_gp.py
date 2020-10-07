"""
A GP prior over qso spectrum for zQSO estimation.
"""
from .null_gp import NullGP

import numpy as np
from scipy import interpolate
import h5py

from .zqso_set_parameters import ZParameters
from .zqso_samples import ZSamples


class ZGP(NullGP):
    """
    GP inference on zQSO:
        p(y | λ, σ², M, ω, blue_σ², red_σ²),
    where we also model the data outside of the modelling window
    as i.i.d Gaussians. See details in
    <Automated Measurement of Quasar Redshift with a Gaussian Process
    https://arxiv.org/abs/2006.07343>

    :param rest_wavelengths: λ, the range of λ you model your GP on QSO emission
    :param mu: mu, the mean model of the GP.
    :param M: M, the low rank decomposition of the covariance kernel: K = MM^T.
    :param bluewards_mu: the mean model for blueward part of the data outside of
        the modelling window.
    :param bluewards_sigma: the noise model for blueward part of the data outside of
        the modelling window.
    :param redwards_mu: the mean model for redward part of the data outside of
        the modelling window.
    :param redwards_sigma: the noise model for redward part of the data outside of
        the modelling window.
    """

    def __init__(
        self,
        params: ZParameters,
        z_qso_samples: ZSamples,
        rest_wavelengths: np.ndarray,
        mu: np.ndarray,
        M: np.ndarray,
        bluewards_mu: float,
        redwards_mu: float,
        bluewards_sigma: float,
        redwards_sigma: float,
    ):
        self.params = params
        self.z_qso_samples = z_qso_samples

        # learned GP model
        self.rest_wavelengths = rest_wavelengths
        self.mu = mu
        self.M = M
        self.bluewards_mu = bluewards_mu
        self.redwards_mu = redwards_mu
        self.bluewards_sigma = bluewards_sigma
        self.redwards_sigma = redwards_sigma

        # preprocess model interpolants
        self.mu_interpolator = interpolate.interp1d(rest_wavelengths, mu)
        self.list_M_interpolators = [
            interpolate.interp1d(rest_wavelengths, eigenvector) for eigenvector in M.T
        ]

    def get_interp(
        self, x: np.ndarray, y: np.ndarray, wavelengths: np.ndarray, z_qso: float
    ) -> None:
        """
        Build and interpolate the GP model onto the observed data.
        
        p(y | λ, zqso, v, ω, M_nodla) = N(y; μ, (K + Ω) + V)
        
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

        The difference between this and the null model set_data is that
        we need to keep all of the data no matter they are inside or
        outside the modelling range, and extend the model to fill up
        all of the data points.

        :param X: (n_points, ) this_wavelengths, the `OBSERVED` wavelengths of the spectrum.
        :param Y: (n_points, ) this_flux, the flux of the observed spectrum
        :param noise_variance: (n_points, ) the instrumental noise variance per pixel.
        :param pixel_mask: (n_points, ) the pixel mask corresponding to the read_spec you used.
        :param z_qso: the redshift of the quasar.

        Note: n_points represents number of pixels within a spectrum
        """
        # variant redshift in quasars
        self.z_qso = z_qso

        # cut-off observations
        max_pos_lambda = self.params.observed_wavelengths(self.params.max_lambda, z_qso)
        min_pos_lambda = self.params.observed_wavelengths(self.params.min_lambda, z_qso)

        max_observed_lambda = np.min((max_pos_lambda, np.max(X)))
        min_observed_lambda = np.max((min_pos_lambda, np.min(X)))

        labmda_observed = max_observed_lambda - min_observed_lambda

        # filter the spectrum; these quantities are within the modelling window
        ind = (X > min_observed_lambda) * (X < max_observed_lambda)
        self.y = Y[ind]
        self.this_wavelengths = X[ind]
        self.v = noise_variance[ind]
        self.pixel_mask = pixel_mask[ind]

        # convert to QSO rest frame
        self.x = self.params.emitted_wavelengths(X[ind], z_qso)

        # normalize flux
        if normalize:
            ind = (self.x >= self.params.normalization_min_lambda) & (
                self.x <= self.params.normalization_max_lambda
            )
            this_median = np.nanmedian(self.y[ind])
            self.y = self.y / this_median
            self.v = self.v / this_median ** 2

        # Normalise the observed flux for out-of-range model since the
        # redward- and blueward- models were trained with normalisation.
        # Find probability for out-of-range model
        this_normalized_flux = (
            Y / this_median
        )  # since we've modified self.y, we thus need to
        # use this_out_flux which is outside the parfor loop
        this_normalized_v = noise_variance / this_median ** 2

        # select blueward region
        ind_bw = (X < min_observed_lambda) & (~pixel_mask)
        self.y_bw = this_normalized_flux[ind_bw]
        self.v_bw = this_normalized_v[ind_bw]
        # select redward region
        ind_rw = (X > max_observed_lambda) & (~pixel_mask)
        self.y_rw = this_normalized_flux[ind_rw]
        self.v_rw = this_normalized_v[ind_rw]

        # apply pixel mask and filter spectrum within modelling range
        ind = (self.x >= self.params.min_lambda) & (self.x <= self.params.max_lambda)
        ind = ind & (~self.pixel_mask)

        self.this_wavelengths = self.this_wavelengths[ind]
        self.x = self.x[ind]
        self.y = self.y[ind]
        self.v = self.v[ind]

        self.v[np.isinf(self.v)] = np.nanmean(self.v)  # rare kludge to fix bad data

        self.ind = ind

        if build_model:
            self.get_interp(self.x, self.y, self.this_wavelengths, self.z_qso)

    def log_model_evidence(self) -> float:
        """
        Compute the log model evidence with the learned model.
        
        Note: assume the model has already interpolated onto the observed data,
        and data got loaded into the GP instance.        
        """
        # log likelihood within the modelling window
        log_likelihood = self.log_mvnpdf_low_rank(
            self.y, self.this_mu, self.this_M, self.v
        )

        n_bw = self.y_bw.shape[0]
        n_rw = self.y_rw.shape[0]

        # calculate log likelihood of iid multivariate normal with
        #   log N(y; mu, diag(V) + sigma^2 )
        bw_log_likelihood = self.log_mvnpdf_iid(
            self.y_bw,
            self.bluewards_mu * np.ones((n_bw,)),
            self.bluewards_sigma ** 2 * np.ones((n_bw,)) + self.v_bw,
        )
        rw_log_likelihood = self.log_mvnpdf_iid(
            self.y_rw,
            self.redwards_mu * np.ones((n_rw,)),
            self.redwards_sigma ** 2 * np.ones((n_rw,)) + self.v_rw,
        )

        return log_likelihood + bw_log_likelihood + rw_log_likelihood

    def inference_z_qso(
        self,
        wavelengths: np.ndarray,
        flux: np.ndarray,
        noise_variance: np.ndarray,
        pixel_mask: np.ndarray,
        z_qso_min: float = 2.14,
        z_qso_max: float = 6.16,
    ):
        """
        Sample the zQSO within a prior volume define in self.z_qso_samples
        """
        sample_log_likelihoods = np.full((self.z_qso_samples.num_zqso_samples,), np.nan)
        sample_z_qsos = self.z_qso_samples.sample_z_qsos(
            z_qso_min=z_qso_min, z_qso_max=z_qso_max
        )

        for i, z_qso in enumerate(sample_z_qsos):
            # set the data and interpolate the model
            self.set_data(
                wavelengths,
                flux,
                noise_variance,
                pixel_mask,
                z_qso=z_qso,
                normalize=True,
                build_model=True,
            )

            sample_log_likelihoods[i] = self.log_model_evidence()

        self.sample_log_likelihoods = sample_log_likelihoods

        # maximum a posteriori
        I = np.nanargmax(sample_log_likelihoods)
        self.z_map = sample_z_qsos[I]
        print("[Info] Z MAP = {:.3g}".format(self.z_map))

    @staticmethod
    def log_mvnpdf_iid(y: np.ndarray, mu: np.ndarray, d: np.ndarray,) -> float:
        """
        computes mutlivariate normal dist with
        each dim is iid, so no covariance. 
            log N(y; mu, diag(d))

        :param y: this_flux, (n_points, )
        :param mu: this_mu, the mean vector of GP, (n_points, )
        :param d: diagonal noise term, (n_points, )
        """
        log_2pi = 1.83787706640934534

        n = d.shape[0]

        y = y[:, None] - mu[:, None]

        d_inv = 1 / d[:, None]  # (n_points, 1)
        D_inv_y = d_inv * y  # (n_points, 1)

        K_inv_y = D_inv_y  # (n_points, 1)

        log_det_K = np.sum(np.log(d))

        log_p = -0.5 * (np.matmul(y.T, K_inv_y).sum() + log_det_K + n * log_2pi)

        return log_p

    @property
    def this_noise(self):
        """
        noise kernel: instrumental noise
        """
        return self.v


class ZGPMAT(ZGP):
    """
    Load learned model from .mat file
    """

    def __init__(
        self,
        params: ZParameters,
        z_qso_samples: ZSamples,
        learned_file: str = "learned_zqso_only_model_outdata_normout_dr9q_minus_concordance_norm_1176-1256.mat",
    ):
        with h5py.File(learned_file, "r") as learned:

            rest_wavelengths = learned["rest_wavelengths"][:, 0]
            mu = learned["mu"][:, 0]
            M = learned["M"][()].T
            bluewards_mu = learned["bluewards_mu"][0, 0]
            redwards_mu = learned["redwards_mu"][0, 0]
            bluewards_sigma = learned["bluewards_sigma"][0, 0]
            redwards_sigma = learned["redwards_sigma"][0, 0]

        super().__init__(
            params=params,
            z_qso_samples=z_qso_samples,
            rest_wavelengths=rest_wavelengths,
            mu=mu,
            M=M,
            bluewards_mu=bluewards_mu,
            redwards_mu=redwards_mu,
            bluewards_sigma=bluewards_sigma,
            redwards_sigma=redwards_sigma,
        )
