''''
Cython log model evidence calculations
'''
import numpy as np
from gpy_dla_detection.null_gp import NullGP
import scipy
from gpy_dla_detection.voigt import voigt_absorption
from libc.math cimport exp as c_exp
# from cython.parallel import prange


# # to retain only unmasked pixels from computed absorption profile
# mask_ind = ~self.pixel_mask[self.ind]

def this_dla_gp(
    double[:] z_dlas,
    double[:] nhis,
    double[:] padded_wavelengths,
    double[:] this_mu,
    double[:, :] this_M,
    double[:] this_omega2,
    long[:] mask_index, # np.where(mask_ind)[0]
    int num_lines,
):
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
    cdef int k_dlas = z_dlas.shape[0]
    
    cdef int length = padded_wavelengths.shape[0] - 6 # 2 * width
    cdef int this_length = this_mu.shape[0]
    cdef int this_k = this_M.shape[1]

    cdef double[:] absorption = np.zeros(length)
    cdef double[:] dla_mu = np.zeros(this_length)
    cdef double[:, :] dla_M = np.zeros((this_length, this_k))
    cdef double[:] dla_omega2 = np.zeros(this_length)
    cdef double[:] this_absorption = np.zeros(this_length)

    cdef int i,j,k


    # absorption corresponding to this sample
    absorption = voigt_absorption(
        padded_wavelengths,
        z_dla=z_dlas[0],
        nhi=nhis[0],
        num_lines=num_lines,
    )

    # absorption corresponding to other DLAs in multiple DLA samples
    for i in range(1, k_dlas):
        absorption2 = voigt_absorption(
            padded_wavelengths,
            z_dla=z_dlas[i],
            nhi=nhis[i],
            num_lines=num_lines,
        )
        for j in range(length):
            absorption[j] = absorption[j] * absorption2[j]

    # this_absorption[:] = absorption[mask_ind]
    for i in range(this_length):
        k = mask_index[i]
        this_absorption[i] = absorption[k]

    for i in range(this_length):
        dla_mu[i] = this_mu[i] * this_absorption[i]
        dla_omega2[i] = this_omega2[i] * this_absorption[i] ** 2

    for i in range(this_length):
        for j in range(this_k):
            dla_M[i, j] = this_M[i, j] * this_absorption[i]

    return dla_mu, dla_M, dla_omega2

def sample_log_likelihood_k_dlas(
    double[:] y,
    double[:] v,
    double[:] z_dlas,
    double[:] nhis,
    double[:] padded_wavelengths,
    double[:] this_mu,
    double[:, :] this_M,
    double[:] this_omega2,
    long[:] mask_index, # np.where(mask_ind)[0]
    int num_lines,
):
    """
    Compute the log likelihood of k DLAs within a quasar spectrum:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, {z_dla, logNHI}_{i=1}^k)

    :param z_dlas: an array of z_dlas you want to condition on
    :param nhis: an array of nhis you want to condition on
    """
    cdef int this_length = this_mu.shape[0]
    cdef int this_k = this_M.shape[1]

    cdef double[:] dla_mu = np.zeros(this_length)
    cdef double[:, :] dla_M = np.zeros((this_length, this_k))
    cdef double[:] dla_omega2 = np.zeros(this_length)

    cdef double[:] d = np.zeros(this_length)

    cdef double sample_log_likelihood

    dla_mu, dla_M, dla_omega2 = this_dla_gp(
        z_dlas,
        nhis,
        padded_wavelengths,
        this_mu,
        this_M, 
        this_omega2,
        mask_index,
        num_lines
    )

    for i in range(this_length):
        d[i] = dla_omega2[i] + v[i]

    sample_log_likelihood = NullGP.log_mvnpdf_low_rank(
        np.asarray(y), np.asarray(dla_mu), np.asarray(dla_M), np.asarray(d)
    )

    return sample_log_likelihood

# the major for loop for QMC sampling
def log_model_evidences(
    double[:] y,
    double[:] v,
    int max_dlas,
    int num_dla_samples,
    double[:] sample_z_dlas,
    double[:] nhi_samples,
    double[:] padded_wavelengths,
    double[:] this_mu,
    double[:, :] this_M,
    double[:] this_omega2,
    long[:] mask_index, # np.where(mask_ind)[0]
    int num_lines,
    double min_z_separation
):
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
    cdef int i, j, k, num_dlas, ind, base_ind
    cdef double occams_factor = np.log(num_dla_samples)
    # prepare the samples to be fed into the sample_log_likelihood function
    cdef double[:] z_dlas = np.zeros((max_dlas, ))
    cdef double[:] nhis = np.zeros((max_dlas, ))
    cdef double z_dla_1, z_dla_2

    # allocate the final log model evidences
    cdef double[:] log_likelihoods_dla = np.empty((max_dlas, )) 
    for i in range(max_dlas):
        log_likelihoods_dla[:] = np.nan

    # base inds to store the QMC samples to be resampled according
    # the prior, which is the posterior of the previous run.
    cdef int[:, :] base_sample_inds = np.zeros((max_dlas - 1, num_dla_samples), dtype=np.intc)

    # sorry, let me follow the convention of the MATLAB code here
    # could be changed to (max_dlas, num_dla_samples) in the future.
    cdef double[:, :] sample_log_likelihoods = np.empty((num_dla_samples, max_dlas))
    for i in range(num_dla_samples):
        for j in range(max_dlas):
            sample_log_likelihoods[i, j] = np.nan

    # compute probabilities under DLA model for each of the sampled
    # (normalized offset, log(N HI)) pairs
    for num_dlas in range(max_dlas):  # count from zero to max_dlas - 1

        # [Need to be parallelized]
        # Roman's code has this part to be parallelized.
        # for i in prange(num_dla_samples, nogil=True):
        for i in range(num_dla_samples):        
            # query the 1st DLA parameter {z_dla, logNHI}_{i=1} from the
            # given DLA samples.
            for j in range(num_dlas + 1):
                if j == 0:
                    z_dlas[j] = sample_z_dlas[i]
                    nhis[j] = nhi_samples[i]
                # query the 2:k DLA parameters {z_dla, logNHI}_{i=2}^k_dlas
                else:
                    base_ind = base_sample_inds[j-1, i]

                    # append to samples to be applied on calculating the log likelihood
                    z_dlas[j] = sample_z_dlas[base_ind]
                    nhis[j] = nhi_samples[base_ind]

            # store the sample log likelihoods conditioned on k-DLAs
            sample_log_likelihoods[i, num_dlas] = sample_log_likelihood_k_dlas(
                y,
                v,
                z_dlas[:(num_dlas+1)],
                nhis[:(num_dlas+1)],
                padded_wavelengths,
                this_mu,
                this_M,
                this_omega2,
                mask_index,
                num_lines                                
            )

        # check if any pair of dlas in this sample is too close this has to
        # happen outside the parfor because "continue" slows things down
        # dramatically
        if num_dlas > 0:
            for i in range(num_dla_samples):
                if find_min_z_separation(
                    i,
                    num_dlas,
                    num_dla_samples,
                    sample_z_dlas,
                    base_sample_inds,
                    min_z_separation
                ):
                    sample_log_likelihoods[i, num_dlas] = np.nan

        # back to use numpy arrays; should be fine since the computationally expensive part
        # is in the above.

        # to prevent numerical underflow
        max_log_likelihood = np.nanmax( np.asarray( sample_log_likelihoods[:, num_dlas]) )

        sample_probabilities = np.exp(
            np.asarray( sample_log_likelihoods[:, num_dlas]) - max_log_likelihood
        )

        log_likelihoods_dla[num_dlas] = (
            max_log_likelihood
            + np.log(np.nanmean(sample_probabilities))
            - occams_factor * (num_dlas + 1) # additional + 1
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

        resampled_base_sample_ind = np.random.choice(
            np.arange(num_dla_samples).astype(np.int32),
            size=num_dla_samples,
            replace=True,
            p=W / W.sum(),
        )

        for i in range(num_dla_samples):
            base_sample_inds[num_dlas, i] = resampled_base_sample_ind[i]

    return log_likelihoods_dla, sample_log_likelihoods

def find_min_z_separation(
    int i,
    int num_dlas,
    int num_dla_samples,
    double[:] sample_z_dlas,
    int[:, :] base_sample_inds,
    double min_z_separation,
):
    cdef int j, k, ind
    cdef double z_dla_1, z_dla_2

    for j in range(num_dlas + 1):
        # query the first DLA location
        if j == 0:
            z_dla_1 = sample_z_dlas[i]
        else:
            ind = base_sample_inds[j-1, i]
            z_dla_1 = sample_z_dlas[ind]

        for k in range(j + 1, num_dlas + 1):
            # query the second DLA location
            ind = base_sample_inds[k-1, i]
            z_dla_2 = sample_z_dlas[ind]

            if np.abs(z_dla_1 - z_dla_2) < min_z_separation:
                return True

    return False
