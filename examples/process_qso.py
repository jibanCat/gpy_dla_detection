"""
A function to process QSOs based on Bayesian model selection
in Ho-Bird-Garnett (2020)
"""

from typing import Union, List, Optional, Any, Tuple

import time
import numpy as np

from gpy_dla_detection import read_spec

from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog

from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT

from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT

from gpy_dla_detection.bayesian_model_selection import BayesModelSelect


def process_qso(
    filename: str,
    z_qso: float,
    read_spec=read_spec.read_spec,
    catalog_file: str = "data/dr12q/processed/catalog.mat",
    learned_file: str = "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
    prior_los: str = "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
    prior_dla: str = "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    dla_samples_file: str = "data/dr12q/processed/dla_samples_a03.mat",
    subdla_samples_file: str = "data/dr12q/processed/subdla_samples.mat",
    max_dlas: int = 4,
    min_z_separation: float = 3000.0,
    broadening: bool = True,
) -> Tuple[BayesModelSelect, List[Any]]:
    """
    Read fits file from qso_list and process each QSO with Bayesian model selection.

    :param qso_list: a list of fits filenames
    :param z_qso_list: a list of zQSO corresponding to the spectra in the qso_list
    :param read_spec: a function to read the fits file.
    :param broadening: whether to implement the instrumental broadening in the SDSS.

    (Comments from the MATLAB code)
    Run DLA detection algorithm on specified objects while using
    lower lognhi range (defined in set_lls_parameters.m) as an
    alternative model; 

    Note: model_posterior(quasar_ind, :) ... 
        = [p(no dla | D), p(lls | D), p(1 dla | D), p(2 dla | D), ...]
    also note that we should treat lls as no dla. 
    
    For higher order of DLA models, we consider the Occam's Razor effect due
    to normalisation in the higher dimensions of the parameter space.
    
    We implement exp(-optical_depth) to the mean-flux
        µ(z) := µ * exp( - τ (1 + z)^β ) ; 
        1 + z = lambda_obs / lambda_lya
    the prior of τ and β are taken from Kim, et al. (2007). 
    
    Nov 18, 2019: add all Lyman series to the effective optical depth
        effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
    where 
        1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)
    
    Dec 25, 2019: add Lyman series to the noise variance training
      s(z)     = 1 - exp(-effective_optical_depth) + c_0 
    
    March 8, 2019: add additional Occam's razor factor between DLA models and null model:
      P(DLAs | D) := P(DLAs | D) / num_dla_samples
    """

    param = Parameters()

    # prepare these files by running the MATLAB scripts until build_catalog.m
    prior = PriorCatalog(param, catalog_file, prior_los, prior_dla,)
    dla_samples = DLASamplesMAT(param, prior, dla_samples_file)

    subdla_samples = SubDLASamplesMAT(param, prior, subdla_samples_file)

    # initialize Bayesian model selection class, with maximum 4 DLAs and at least 1 subDLA,
    # which is the same as Ho-Bird-Garnett (2020).
    bayes = BayesModelSelect([0, 1, max_dlas], 2)  # 0 DLA for null; 1 subDLA; 4 DLAs.
    # 2 for the location of the DLA model in the list

    tic = time.time()

    np.random.seed(0)

    # the read spec function must return these four ndarray in this order
    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
    rest_wavelengths = param.emitted_wavelengths(wavelengths, z_qso)

    # Null model GP : a GP model without any DLA intervening, we also don't need to
    # run QMC sampling on this model.
    gp = NullGPMAT(param, prior, learned_file,)

    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # DLA Model GP : this is null model GP + Voigt profile, which is parameterised with
    # {(z_dla, logNHI)}_{i=1}^{k} parameters. k stands for maximum DLA we want to model.
    # we will estimate log posteriors for DLA(1), ..., DLA(k) models.
    dla_gp = DLAGPMAT(
        params=param,
        prior=prior,
        dla_samples=dla_samples,
        min_z_separation=3000.0,
        learned_file=learned_file,
        broadening=broadening,
    )
    dla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    subdla_gp = SubDLAGPMAT(
        params=param,
        prior=prior,
        dla_samples=subdla_samples,
        min_z_separation=3000.0,
        learned_file=learned_file,
        broadening=broadening,
    )
    subdla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # append and get the model list for Bayesian model comparison
    model_list = [gp, subdla_gp, dla_gp]

    # run bayesian model selection and get log posteriors
    log_posteriors = bayes.model_selection(model_list, z_qso)
    # at the same time, log_priors, log_likelihoods, log_posteriors will be saved to
    # the bayes instance.
    print("[Info] log posterior:", log_posteriors)
    print("[Info] model posterior:", bayes.model_posteriors)

    toc = time.time()
    # very time consuming: ~ 4 mins for a single spectrum without parallelized.
    print("[Info] spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    return bayes, model_list