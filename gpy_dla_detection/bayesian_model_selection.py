from typing import List, Union
from itertools import chain
import warnings
import numpy as np
from scipy.special import logsumexp
from .null_gp import NullGP
from .dla_gp import DLAGP
from .subdla_gp import SubDLAGP


class BayesModelSelect:
    """
    Bayesian model selection:

        p(M | D) = P(M) * P(D | M) / ∑_i( P(M_i) * P(D | M_i) )

    :attr all_max_dlas: List of maximum DLAs for each model in the model list.
    :attr dla_model_ind: Index of the DLA model in the model list.
    """

    def __init__(self, all_max_dlas: List[int] = [0, 1, 4], dla_model_ind: int = 2):
        """
        Initialize the Bayesian model selection with the list of models and their maximum DLAs.

        :param all_max_dlas: A list of integers, indicating the number of DLAs for each model.
                             0 indicates the null model.
        :param dla_model_ind: The index of the DLA model in the list. Default is 2.
        """
        self.all_max_dlas = all_max_dlas
        self.dla_model_ind = dla_model_ind

    def model_selection(
        self,
        model_list: List[Union[NullGP, SubDLAGP, DLAGP]],
        z_qso: float,
        max_workers: int = None,
        batch_size: int = 100,
        pool=None,
    ) -> np.ndarray:
        """
        Perform Bayesian model selection for a list of models.

        :param model_list: List of models to compare (NullGP, SubDLAGP, DLAGP).
        :param z_qso: The redshift of the quasar.
        :param max_workers: Number of workers to use for parallel processing.
        :param batch_size: Batch size for parallel computation.
        :param pool: Pool object for parallel processing.

        :return: Array of log posteriors for each model.
        """
        assert isinstance(model_list[0], NullGP)  # Null model
        assert isinstance(model_list[-1], DLAGP)  # DLA model
        assert len(model_list) > self.dla_model_ind  # Check if DLA model is in the list

        log_posteriors = []
        log_priors = []
        log_likelihoods = []

        # Prepare the model priors for each model
        for i, num_dlas in enumerate(self.all_max_dlas):
            # Skip the null model prior (no DLAs)
            if num_dlas == 0:
                log_priors.append([np.nan])  # Null model prior
                continue

            # Compute the model priors for DLA and subDLA models
            log_priors_dla = model_list[i].log_priors(z_qso, num_dlas)
            log_priors.append(log_priors_dla)

        # Calculate the null model prior as (1 - sum of all other model priors)
        log_priors = np.array(list(chain(*log_priors)))
        log_priors[0] = np.log(1 - np.exp(logsumexp(log_priors[1:])))

        # Calculate model evidences (log likelihoods)
        # TODO: Here is the place if you want to stop the for loop when you are sure that no dla is present
        for i, num_dlas in enumerate(self.all_max_dlas):
            if num_dlas == 0:
                # Null model evidence
                log_likelihood_no_dla = model_list[i].log_model_evidence()
                log_likelihoods.append([log_likelihood_no_dla])
            else:
                # Use parallel model evidence calculation for DLA models
                log_likelihoods_dla = model_list[i].parallel_log_model_evidences(
                    num_dlas,
                    max_workers=max_workers,
                    batch_size=batch_size,
                    executor=pool,
                )
                log_likelihoods.append(log_likelihoods_dla)

        # Flatten the log likelihoods and compute the posteriors
        log_likelihoods = np.array(list(chain(*log_likelihoods)))
        log_posteriors = log_likelihoods + log_priors

        # Perform the tolerance check
        if (
            not np.isfinite(log_likelihoods[2])
            or not np.isfinite(log_priors[2])
            or not np.isfinite(log_posteriors[2])
        ):
            warnings.warn(
                f"Invalid values encountered in log likelihoods, priors, or posteriors."
            )
        else:
            difference = np.abs(
                (log_likelihoods[2] + log_priors[2]) - log_posteriors[2]
            )
            if difference >= 1e-4:
                warnings.warn(f"Posterior mismatch detected: difference = {difference}")

        # Store results for later use
        self.log_priors = log_priors
        self.log_likelihoods = log_likelihoods
        self.log_posteriors = log_posteriors

        return log_posteriors

    @property
    def dla_model_posterior_ind(self):
        """
        Find the index for DLA model posteriors in the log_posteriors array.

        Default is [no DLA, subDLA, 1 DLA, 2 DLA, 3 DLA, 4 DLA],
        corresponding to all_max_dlas = [0, 1, 4].
        """
        ind = np.zeros((self.log_posteriors.shape[0],), dtype=np.bool_)
        ind[-self.all_max_dlas[self.dla_model_ind] :] = True

        self._dla_model_posterior_ind = ind

        return self._dla_model_posterior_ind

    @property
    def model_posteriors(self):
        """
        Compute the model posteriors as normalized probabilities.
        """
        indisnan = np.isnan(self.log_posteriors)
        sum_log_posteriors = logsumexp(self.log_posteriors[~indisnan])
        return np.exp(self.log_posteriors - sum_log_posteriors)

    @property
    def model_evidences(self):
        """
        Compute the model evidences as normalized probabilities.
        """
        indisnan = np.isnan(self.log_likelihoods)
        sum_log_evidences = logsumexp(self.log_likelihoods[~indisnan])
        return np.exp(self.log_likelihoods - sum_log_evidences)

    @property
    def model_priors(self):
        """
        Compute the model priors as normalized probabilities.
        """
        indisnan = np.isnan(self.log_priors)
        sum_log_priors = logsumexp(self.log_priors[~indisnan])
        return np.exp(self.log_priors - sum_log_priors)

    @property
    def p_dla(self):
        """
        Compute the posterior probability of having a DLA.
        """
        model_posteriors = self.model_posteriors
        self._p_dla = np.nansum(model_posteriors[self.dla_model_posterior_ind])
        return self._p_dla

    @property
    def p_no_dla(self):
        """
        Compute the posterior probability of not having a DLA.
        """
        return 1 - self.p_dla
