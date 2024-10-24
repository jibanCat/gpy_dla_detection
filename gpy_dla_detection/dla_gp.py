"""
A GP class for having multiple DLAs intervening in a given slightline. 
"""

from typing import Tuple, Optional, Callable, List
import os

import concurrent.futures
from concurrent.futures import as_completed


import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
import h5py

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .null_gp import NullGP

# Attempt to import VoigtProfile from voigt_fast, and fall back to voigt if it fails
try:
    from .voigt_fast import VoigtProfile

    voigt_absorption = VoigtProfile().compute_voigt_profile
# OSError, ImportError:
except (OSError, ImportError):
    from .voigt import voigt_absorption

# this could be replaced to DLASamples in the future;
# I import this is for the convenient of my autocomplete
from .dla_samples import DLASamplesMAT

# Limit the number of workers to the number of CPU cores
# max_workers = os.cpu_count() * 2


def process_sample(
    i: int,
    num_dlas: int,
    sample_z_dlas: np.ndarray,
    base_sample_inds: np.ndarray,
    dla_samples: DLASamplesMAT,
    params: Parameters,
    sample_log_likelihood_k_dlas: Callable[[np.ndarray, np.ndarray], float],
    min_z_separation: float,
) -> float:
    """
    Process a single sample by querying DLA parameters and computing the log likelihood.

    This function retrieves the DLA parameters (redshift `z_dlas` and column density `logNHI`) for the
    current sample. If `num_dlas` > 0, it retrieves additional parameters for multiple DLAs. Finally, it
    computes the log likelihood of the k-DLA model for the given sample.

    Args:
        i (int): Index of the current sample.
        num_dlas (int): Number of DLAs in the model for this sample.
        sample_z_dlas (np.ndarray): Array of sampled redshift values for DLAs.
        base_sample_inds (np.ndarray): Base indices to be resampled according to the prior.
        dla_samples ('DLASamplesMAT'): Object containing the DLA sample catalog.
        params ('Parameters'): Model parameters object.
        sample_log_likelihood_k_dlas (Callable): Function to compute the log likelihood of k-DLA model.
        min_z_separation (float): Minimum redshift separation for the DLA samples.

    Returns:
        float: The computed log likelihood for this sample.
    """

    # Query the 1st DLA parameter {z_dla, logNHI}_{i=1} from the given DLA samples
    z_dlas = np.array([sample_z_dlas[i]])
    log_nhis = np.array([dla_samples.log_nhi_samples[i]])
    nhis = np.array([dla_samples.nhi_samples[i]])

    # Query the 2:k DLA parameters {z_dla, logNHI}_{i=2}^k_dlas
    if num_dlas > 0:
        base_ind = base_sample_inds[:num_dlas, i]
        z_dlas_2_k = sample_z_dlas[base_ind]
        log_nhis_2_k = dla_samples.log_nhi_samples[base_ind]
        nhis_2_k = dla_samples.nhi_samples[base_ind]

        # Append to samples to be applied on calculating the log likelihood
        z_dlas = np.append(z_dlas, z_dlas_2_k)
        log_nhis = np.append(log_nhis, log_nhis_2_k)
        nhis = np.append(nhis, nhis_2_k)

        del z_dlas_2_k, log_nhis_2_k, nhis_2_k

    # Compute the sample log likelihoods conditioned on k-DLAs
    log_likelihood = sample_log_likelihood_k_dlas(z_dlas, nhis) - np.log(
        params.num_dla_samples
    )

    return log_likelihood


def process_batch(
    batch_indices: List[int],
    num_dlas: int,
    sample_z_dlas: np.ndarray,
    base_sample_inds: np.ndarray,
    dla_samples: DLASamplesMAT,
    params: Parameters,
    sample_log_likelihood_k_dlas: callable,
    min_z_separation: float,  # Add min_z_separation as an argument
) -> List[float]:
    """
    Process a batch of samples. For each sample in the batch, this function computes
    the log likelihood using `process_sample` and returns the results as a list.

    Args:
        batch_indices (List[int]): Indices of the samples in the batch.
        num_dlas (int): Number of DLAs to consider in the model.
        sample_z_dlas (np.ndarray): Array of sampled redshift values for DLAs.
        base_sample_inds (np.ndarray): Base indices for resampling according to the prior.
        dla_samples ('DLASamplesMAT'): Object containing the DLA sample catalog.
        params ('Parameters'): Model parameters object.
        sample_log_likelihood_k_dlas (callable): Function to compute log likelihood for each sample.
        min_z_separation (float): Minimum redshift separation for DLA pairs.

    Returns:
        List[float]: List of log likelihoods for each sample in the batch.
    """
    batch_results = []  # This will store the results for the entire batch

    # Loop through each sample index in the batch and process it
    for i in batch_indices:
        # Process each sample using the same logic as process_sample
        result = process_sample(
            i,
            num_dlas,
            sample_z_dlas,
            base_sample_inds,
            dla_samples,
            params,
            sample_log_likelihood_k_dlas,
            min_z_separation,  # Pass the missing argument
        )
        batch_results.append(result)  # Store the result in the batch result list

    return batch_results  # Return the list of results for the batch


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
        broadening: bool = True,
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

        self.broadening = broadening

        # Initialize a cache for Voigt profiles
        self.voigt_cache = {}

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
            (
                max_dlas - 1,
                self.params.num_dla_samples,
            ),
            dtype=np.int32,
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

    def parallel_log_model_evidences(
        self,
        max_dlas: int,
        max_workers: int = 32,
        batch_size: int = 313,
        executor=None,
    ) -> np.ndarray:
        """
        Parallelized version of the log model evidences computation using process-based parallelization.

        This method computes the log likelihoods of the k-DLA models in parallel using `ProcessPoolExecutor`.
        The process is repeated for each number of DLAs (up to `max_dlas`), and the results are stored
        in an array.

        Args:
            max_dlas (int): The maximum number of DLAs to be considered in the model.
            max_workers (int, optional): Maximum number of workers to use. Defaults to number of CPU cores * 2.
            batch_size (int, optional): Number of samples per batch. Defaults to 100.
            executor (ProcessPoolExecutor, optional): An existing executor to reuse; if not provided, a new one is created.

        Returns:
            np.ndarray: Array containing the computed log likelihoods for 1 to `max_dlas` DLAs.
        """
        # Set default number of workers if not provided
        if max_workers is None:
            max_workers = os.cpu_count() * 2

        # Allocate the final log model evidences
        log_likelihoods_dla = np.empty((max_dlas,))
        log_likelihoods_dla[:] = np.nan

        # Base indices to store the QMC samples to be resampled according to the prior
        base_sample_inds = np.zeros(
            (max_dlas - 1, self.params.num_dla_samples), dtype=np.int32
        )

        # Allocate sample log likelihoods array
        sample_log_likelihoods = np.empty((self.params.num_dla_samples, max_dlas))
        sample_log_likelihoods[:] = np.nan

        # Prepare z_dla samples
        sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )

        # Create batches of indices
        indices = list(range(self.params.num_dla_samples))
        batches = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        # Use an external executor if provided, otherwise create a new one
        executor_is_external = executor is not None
        if not executor_is_external:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        try:
            for num_dlas in range(max_dlas):  # Iterate from 0 to max_dlas - 1
                # Submit the tasks for each batch to the executor
                futures = {
                    executor.submit(
                        process_batch,
                        batch,
                        num_dlas,
                        sample_z_dlas,
                        base_sample_inds,
                        self.dla_samples,
                        self.params,
                        self.sample_log_likelihood_k_dlas,
                        self.min_z_separation,
                    ): batch
                    for batch in batches
                }

                # Process the results as each batch completes
                for future in as_completed(futures):
                    batch_indices = futures[future]  # Retrieve the batch indices
                    batch_results = future.result()  # Get the results for this batch

                    # Store the results in the corresponding places in sample_log_likelihoods
                    for i, result in zip(batch_indices, batch_results):
                        sample_log_likelihoods[i, num_dlas] = result

                # Handle NaN values and resampling logic
                if num_dlas > 0:
                    ind = base_sample_inds[:num_dlas, :]
                    all_z_dlas = np.concatenate(
                        [sample_z_dlas[None, :], sample_z_dlas[ind]], axis=0
                    )
                    ind = np.any(
                        np.diff(np.sort(all_z_dlas, axis=0), axis=0)
                        < self.min_z_separation,
                        axis=0,
                    )
                    sample_log_likelihoods[ind, num_dlas] = np.nan

                # Compute the log likelihood for each number of DLAs
                max_log_likelihood = np.nanmax(sample_log_likelihoods[:, num_dlas])
                sample_probabilities = np.exp(
                    sample_log_likelihoods[:, num_dlas] - max_log_likelihood
                )
                log_likelihoods_dla[num_dlas] = (
                    max_log_likelihood
                    + np.log(np.nanmean(sample_probabilities))
                    - np.log(self.params.num_dla_samples) * num_dlas
                )

                if (num_dlas + 1) == max_dlas or np.isnan(
                    log_likelihoods_dla[num_dlas]
                ):
                    break

                # Resampling logic to update base sample indices
                nanind = np.isnan(sample_probabilities)
                W = sample_probabilities
                W[nanind] = 0.0

                base_sample_inds[num_dlas, :] = np.random.choice(
                    np.arange(self.params.num_dla_samples).astype(np.int32),
                    size=self.params.num_dla_samples,
                    replace=True,
                    p=W / W.sum(),
                )

        finally:
            # Shut down the executor if it was created locally
            if not executor_is_external:
                executor.shutdown()

        # Store results for future use
        self.sample_log_likelihoods = sample_log_likelihoods
        self.base_sample_inds = base_sample_inds

        return log_likelihoods_dla

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

        # Reuse or cache the Voigt profiles
        dla_mu, dla_M, dla_omega2 = self.this_dla_gp(z_dlas, nhis)

        # Compute the log-likelihood
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

        # [broadening] use the padded wavelengths for convolution
        # otherwise, should use unmasked wavelengths.
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        # Initialize the absorption profile for the DLA model
        absorption = np.ones(self.unmasked_wavelengths.shape[0])

        # Loop through each DLA and compute/reuse the Voigt profiles
        for j in range(k_dlas):
            # Create a unique cache key for the current (z_dla, nhi) pair
            cache_key = (z_dlas[j], nhis[j], self.broadening)

            if cache_key in self.voigt_cache:
                # Retrieve from cache if available
                cached_absorption = self.voigt_cache[cache_key]
            else:
                # Otherwise, compute the Voigt profile and store in cache
                cached_absorption = voigt_absorption(
                    wavelengths,
                    z_dla=z_dlas[j],
                    nhi=nhis[j],
                    num_lines=self.params.num_lines,
                )
                self.voigt_cache[cache_key] = cached_absorption

            # Multiply the absorption profiles for all DLAs
            absorption *= cached_absorption

        absorption = absorption[mask_ind]

        assert len(absorption) == len(self.this_mu)

        dla_mu = self.this_mu * absorption
        dla_M = self.this_M * absorption[:, None]
        dla_omega2 = self.this_omega2 * absorption**2

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

        p_dlas = (this_num_dlas / this_num_quasars) ** np.arange(1, max_dlas + 1)

        for i in range(max_dlas - 1):
            p_dlas[i] = p_dlas[i] - p_dlas[i + 1]

        log_priors_dla = np.log(p_dlas)

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
        broadening: bool = True,
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
            dla_samples,
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
