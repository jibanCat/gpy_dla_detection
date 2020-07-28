"""
bayesian_model_selection.py : A class to perform DLA classification
using Bayes rule via Bayesian model selection (or known to be
Bayesian hypothesis testing)

Check Roman's Lecture 7: https://www.cse.wustl.edu/~garnett/cse515t/spring_2019/
or Mackay's information theory, Chapter 28.
"""
from typing import List, Tuple, Union

from itertools import chain

import numpy as np
from scipy.special import logsumexp

from .null_gp import NullGP
from .dla_gp import DLAGP
from .subdla_gp import SubDLAGP


class BayesModelSelect:
    """
    Bayesian model selection:

        p(M | D) = P(M) * P(D | M) / ∑_i( P(M_i) * P(D | M_i) )
    
    which reads:
    
        model posterior = model prior * model evidence
            / (sum of the model posteriors of all possible models)

    :attr model_list: a List of models we want to compute in Bayesian model selection.
    :attr all_max_dlas: a List of integers indicates number of DLAs to be computed
        for each model in the List. 0 for no DLA, which means NullGP, for max_dlas > 0,
        model evidences will be calculated from .dla_gp.DLAGP.log_model_evidences(max_dlas).
    :attr dla_model_ind: an integer indicates the index of DLA model in the model_list. This
        means all other models within model_list will be considered to be 
        Default is 2.
    """

    def __init__(
        self, all_max_dlas: List[int] = [0, 1, 4], dla_model_ind: int = 2,
    ):
        # a list of models, all have a base class of NullGP
        self.all_max_dlas = all_max_dlas
        self.dla_model_ind = dla_model_ind

    def model_selection(
        self, model_list: List[ Union[NullGP, SubDLAGP, DLAGP] ], z_qso: float
    ) -> np.ndarray:
        """
        Calculate the log model evidences and priors for each model
        in the model_list.

        Default assumption is [null model, subDLA model, DLA model].
        And always assume the first model is null model and the last one is DLA model. 
        """
        assert ~isinstance(model_list[0], DLAGP)
        assert isinstance(model_list[-1], DLAGP)
        assert isinstance(model_list[-1], NullGP)
        assert len(model_list) > self.dla_model_ind

        log_posteriors = []
        log_priors = []
        log_likelihoods = []

        # prepare the model priors first, so we can get the null model prior
        for i, num_dlas in enumerate(self.all_max_dlas):
            # skip null model prior
            if num_dlas == 0:
                log_priors.append([np.nan])
                continue

            # model priors
            log_priors_dla = model_list[i].log_priors(z_qso, num_dlas)
            log_priors.append(log_priors_dla)

        # null model prior is (1 - other model priors)
        log_priors = np.array(list(chain(*log_priors)))
        log_priors[0] = np.log(1 - np.exp(logsumexp(log_priors[1:])))

        # calculating model evidences
        for i, num_dlas in enumerate(self.all_max_dlas):
            # if this is null model
            if num_dlas == 0:
                # model evidence
                log_likelihood_no_dla = model_list[i].log_model_evidence()
                log_likelihoods.append([log_likelihood_no_dla])

                log_posteriors.append([log_likelihood_no_dla + log_priors[i]])

            # if this is for DLA model or subDLA model
            else:
                # model evidence
                log_likelihoods_dla = model_list[i].log_model_evidences(num_dlas)
                log_likelihoods.append(log_likelihoods_dla)

                # model posteriors : this is a numpy array
                log_posteriors.append(log_likelihoods_dla + log_priors[i])

        # flatten the nested list : this is due to each element
        log_likelihoods = np.array(list(chain(*log_likelihoods)))
        log_posteriors = np.array(list(chain(*log_posteriors)))

        self.log_priors = log_priors
        self.log_likelihoods = log_likelihoods
        self.log_posteriors = log_posteriors

        return log_posteriors

    @property
    def dla_model_posterior_ind(self):
        """
        Find the ind for DLA model posteriors in the log_posteriors array.

        Default is [no DLA, subDLA, 1 DLA, 2 DLA, 3 DLA, 4 DLA],
        corresponding to all_max_dlas = [0, 1, 4].
        """
        ind = np.zeros((self.log_posteriors.shape[0],), dtype=np.bool_)
        ind[-self.all_max_dlas[self.dla_model_ind] :] = True

        self._dla_model_posterior_ind = ind

        return self._dla_model_posterior_ind

    @property
    def model_posteriors(self):
        sum_log_posteriors = logsumexp(self.log_posteriors)
        return np.exp(self.log_posteriors - sum_log_posteriors)

    @property
    def model_evidences(self):
        sum_log_evidences = logsumexp(self.log_likelihoods)
        return np.exp(self.log_likelihoods - sum_log_evidences)

    @property
    def model_priors(self):
        sum_log_priors = logsumexp(self.log_priors)
        return np.exp(self.log_priors - sum_log_priors)

    @property
    def p_dla(self):
        model_posteriors = self.model_posteriors
        self._p_dla = np.sum(model_posteriors[self.dla_model_posterior_ind])
        return self._p_dla

    @property
    def p_no_dla(self):
        return 1 - self.p_dla
