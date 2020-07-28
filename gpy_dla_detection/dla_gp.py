"""
A GP class for having multiple DLAs intervening in a given slightline. 
"""
from typing import Tuple, Optional

import numpy as np
import h5py

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .null_gp import NullGP
from .voigt import voigt_absorption

# this could be replaced to DLASamples in the future;
# I import this is for the convenient of my autocomplete
from .dla_samples import DLASamplesMAT

from profilehooks import profile


class DLAGP(NullGP):
    """
    DLA GP model for QSO emission + DLA intervening:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, z_dla, logNHI)

    additional two parameters (z_dla, logNHI) will control the position
    and the strength of the absorption intervening on the QSO emission.

    Since the integration is not tractable, so we use QMC to approximate
    the model evidence. 

    How many QMC samples will be defined in Parameters and DLASamples.

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

    Future: MCMC embedded in the class as an instance method.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: DLASamplesMAT,
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
    ):
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
            prev_tau_0,
            prev_beta,
        )

        self.min_z_separation = self.params.kms_to_z(min_z_separation)

        self.dla_samples = dla_samples

    def log_model_evidences(self, max_dlas: int) -> np.ndarray:
        """
        marginalize out the DLA parameters, {(z_dla_i, logNHI_i)}_{i=1}^k_dlas,
        and return an array of log_model_evidences for 1:k DLA models
        
        Note: we provide an integration method here to reproduce the functionality
        in Ho-Bird-Garnett's code, but we encourage users to improve this sampling
        scheme to be more efficient with another external script by calling
        self.sample_log_likelihood_k_dlas directly.

        :param max_dlas: the number of DLAs we want to marginalise

        :return: [P(D | 1 DLA), ..., P(D | k DLAs)]
        """
        # allocate the final log model evidences
        log_likelihoods_dla = np.empty((max_dlas,))
        log_likelihoods_dla[:] = np.nan

        # base inds to store the QMC samples to be resampled according
        # the prior, which is the posterior of the previous run.
        base_sample_inds = np.zeros(
            (max_dlas - 1, self.params.num_dla_samples,), dtype=np.int32
        )

        # sorry, let me follow the convention of the MATLAB code here
        # could be changed to (max_dlas, num_dla_samples) in the future.
        sample_log_likelihoods = np.empty((self.params.num_dla_samples, max_dlas))
        sample_log_likelihoods[:] = np.nan

        # prepare z_dla samples
        sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )

        # compute probabilities under DLA model for each of the sampled
        # (normalized offset, log(N HI)) pairs
        for num_dlas in range(max_dlas):  # count from zero to max_dlas - 1

            # [Need to be parallelized]
            # Roman's code has this part to be parallelized.
            for i in range(self.params.num_dla_samples):
                # query the 1st DLA parameter {z_dla, logNHI}_{i=1} from the
                # given DLA samples.
                z_dlas = np.array([sample_z_dlas[i]])
                log_nhis = np.array([self.dla_samples.log_nhi_samples[i]])
                nhis = np.array([self.dla_samples.nhi_samples[i]])

                # query the 2:k DLA parameters {z_dla, logNHI}_{i=2}^k_dlas
                if num_dlas > 0:
                    base_ind = base_sample_inds[:num_dlas, i]

                    z_dlas_2_k = sample_z_dlas[base_ind]
                    log_nhis_2_k = self.dla_samples.log_nhi_samples[base_ind]
                    nhis_2_k = self.dla_samples.nhi_samples[base_ind]

                    # append to samples to be applied on calculating the log likelihood
                    z_dlas = np.append(z_dlas, z_dlas_2_k)
                    log_nhis = np.append(log_nhis, log_nhis_2_k)
                    nhis = np.append(nhis, nhis_2_k)

                    del z_dlas_2_k, log_nhis_2_k, nhis_2_k

                # store the sample log likelihoods conditioned on k-DLAs
                sample_log_likelihoods[i, num_dlas] = self.sample_log_likelihood_k_dlas(
                    z_dlas, nhis
                ) - np.log(
                    self.params.num_dla_samples
                )  # additional occams razor

            # check if any pair of dlas in this sample is too close this has to
            # happen outside the parfor because "continue" slows things down
            # dramatically
            if num_dlas > 0:
                # all_z_dlas : (num_dlas, num_dla_samples)
                ind = base_sample_inds[:num_dlas, :]  # (num_dlas - 1, num_dla_samples)

                all_z_dlas = np.concatenate(
                    [sample_z_dlas[None, :], sample_z_dlas[ind]], axis=0
                )  # (num_dlas, num_dla_samples)

                ind = np.any(
                    np.diff(np.sort(all_z_dlas, axis=0), axis=0)
                    < self.min_z_separation,
                    axis=0,
                )
                sample_log_likelihoods[ind, num_dlas] = np.nan

            # to prevent numerical underflow
            max_log_likelihood = np.nanmax(sample_log_likelihoods[:, num_dlas])

            sample_probabilities = np.exp(
                sample_log_likelihoods[:, num_dlas] - max_log_likelihood
            )

            log_likelihoods_dla[num_dlas] = (
                max_log_likelihood
                + np.log(np.nanmean(sample_probabilities))
                - np.log(self.params.num_dla_samples) * num_dlas
            )  # occams razor for more DLA parameters

            # no needs for re-sample the QMC samples for the last run
            if (num_dlas + 1) == max_dlas:
                break

            # if p(D | z_QSO, k DLA) is NaN, then
            # finish the loop.
            # It's usually because p(D | z_QSO, no DLA) is very high, so
            # the higher order DLA model likelihoods already underflowed
            if np.isnan(log_likelihoods_dla[num_dlas]):
                print(
                    "Finish the loop earlier because NaN value in log p(D | z_QSO, {} DLAs)".format(
                        num_dlas
                    )
                )
                break

            # avoid nan values in the randsample weights
            nanind = np.isnan(sample_probabilities)
            W = sample_probabilities
            W[nanind] = 0.0

            base_sample_inds[num_dlas, :] = np.random.choice(
                np.arange(self.params.num_dla_samples).astype(np.int32),
                size=self.params.num_dla_samples,
                replace=True,
                p=W / W.sum(),
            )

        # store sample likelihoods for MAP value calculation
        # this could cause troubles for parallelization in the future
        self.sample_log_likelihoods = sample_log_likelihoods
        self.base_sample_inds = base_sample_inds

        return log_likelihoods_dla

    @profile
    def sample_log_likelihood_k_dlas(
        self, z_dlas: np.ndarray, nhis: np.ndarray
    ) -> float:
        """
        Compute the log likelihood of k DLAs within a quasar spectrum:
            p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, {z_dla, logNHI}_{i=1}^k)

        :param z_dlas: an array of z_dlas you want to condition on
        :param nhis: an array of nhis you want to condition on
        """
        assert len(z_dlas) == len(nhis)

        dla_mu, dla_M, dla_omega2 = self.this_dla_gp(z_dlas, nhis)

        sample_log_likelihood = self.log_mvnpdf_low_rank(
            self.y, dla_mu, dla_M, dla_omega2 + self.v
        )

        return sample_log_likelihood

    def this_dla_gp(
        self, z_dlas: np.ndarray, nhis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the DLA GP model with k intervening DLA profiles onto
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
        assert len(z_dlas) == len(nhis)

        k_dlas = len(z_dlas)

        # to retain only unmasked pixels from computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # absorption corresponding to this sample
        absorption = voigt_absorption(
            self.padded_wavelengths,
            z_dla=z_dlas[0],
            nhi=nhis[0],
            num_lines=self.params.num_lines,
        )

        # absorption corresponding to other DLAs in multiple DLA samples
        for j in range(1, k_dlas):
            absorption = absorption * voigt_absorption(
                self.padded_wavelengths,
                z_dla=z_dlas[j],
                nhi=nhis[j],
                num_lines=self.params.num_lines,
            )

        absorption = absorption[mask_ind]

        assert len(absorption) == len(self.this_mu)

        dla_mu = self.this_mu * absorption
        dla_M = self.this_M * absorption[:, None]
        dla_omega2 = self.this_omega2 * absorption ** 2

        return dla_mu, dla_M, dla_omega2

    def log_priors(self, z_qso: float, max_dlas: int) -> float:
        """
        get the model prior of null model, this is defined to be:
            P(k DLA | zQSO) = P(at least k DLAs | zQSO) - P(at least (k + 1) DLAs | zQSO),
        
        where

            P(at least 1 DLA | zQSO) = M / N
        
        M : number of DLAs below this zQSO
        N : number of quasars below this zQSO

        and

            P(at least k DLA | zQSO) = (M / N)^k
        
        Note: I did not overwrite the NullGP log prior, name of this method is log_prior's'
        for multi-DLAs
        """
        this_num_dlas, this_num_quasars = self.prior.less_ind(z_qso)

        log_priors_dla = np.zeros((max_dlas,))

        p_dlas = (this_num_dlas / this_num_quasars) ** np.arange(1, max_dlas + 1)

        for i in range(max_dlas):
            p_dlas[i] = p_dlas[i] - np.sum(p_dlas[(i + 1) :])

            log_priors_dla[i] = np.log(p_dlas[i])

        return log_priors_dla

    def maximum_a_posteriori(self):
        """
        Find the maximum a posterior parameter pair {(z_dla, logNHI)}_{i=1}^k.

        :return (MAP_z_dla, MAP_log_nhi): shape for each is (max_dlas, max_dlas),
            the 0 dimension is for DLA(k) model and the 1 dimension is for
            the MAP estimates.
        """
        maxinds = np.nanargmax(self.sample_log_likelihoods, axis=0)

        max_dlas = self.sample_log_likelihoods.shape[1]

        MAP_z_dla = np.empty((max_dlas, max_dlas))
        MAP_log_nhi = np.empty((max_dlas, max_dlas))
        MAP_z_dla[:] = np.nan
        MAP_log_nhi[:] = np.nan

        # prepare z_dla samples
        sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )

        for num_dlas, maxind in enumerate(maxinds):
            # store k MAP estimates for DLA(k) model
            if num_dlas > 0:
                # all_z_dlas : (num_dlas, num_dla_samples)
                ind = self.base_sample_inds[
                    :num_dlas, maxind
                ]  # (num_dlas - 1, num_dla_samples)

                MAP_z_dla[num_dlas, : (num_dlas + 1)] = np.concatenate(
                    [[sample_z_dlas[maxind]], sample_z_dlas[ind]]
                )  # (num_dlas, )
                MAP_log_nhi[num_dlas, : (num_dlas + 1)] = np.concatenate(
                    [
                        [self.dla_samples.log_nhi_samples[maxind]],
                        self.dla_samples.log_nhi_samples[ind],
                    ]
                )
            # for DLA(1) model, only store one MAP estimate
            else:
                MAP_z_dla[num_dlas, 0] = sample_z_dlas[maxind]
                MAP_log_nhi[num_dlas, 0] = self.dla_samples.log_nhi_samples[maxind]

        return MAP_z_dla, MAP_log_nhi


class DLAGPMAT(DLAGP):
    """
    Load learned model from .mat file
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: DLASamplesMAT,
        min_z_separation: float = 3000.0,
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
            dla_samples,
            rest_wavelengths,
            mu,
            M,
            log_omega,
            log_c_0,
            log_tau_0,
            log_beta,
            prev_tau_0=0.0023,
            prev_beta=3.65,
            min_z_separation=min_z_separation,
        )
