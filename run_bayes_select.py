"""
Run Bayesian model selection on a given spectrum
using Ho-Bird-Garnett code in Python version.
"""
from typing import Union, List, Optional

import time
import numpy as np
import h5py

from gpy_dla_detection import read_spec

from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog

from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT

from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT

from gpy_dla_detection.bayesian_model_selection import BayesModelSelect

import argparse


def process_qso(
    qso_list: List,
    z_qso_list: List,
    read_spec=read_spec.read_spec,
    max_dlas: int = 4,
):
    """
    Read fits file from qso_list and process each QSO with Bayesian model selection.

    :param qso_list: a list of fits filenames
    :param z_qso_list: a list of zQSO corresponding to the spectra in the qso_list
    :param read_spec: a function to read the fits file.

    :return 

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
    prior = PriorCatalog(
        param,
        "data/dr12q/processed/catalog.mat",
        "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    )
    dla_samples = DLASamplesMAT(
        param, prior, "data/dr12q/processed/dla_samples_a03.mat"
    )

    subdla_samples = SubDLASamplesMAT(
        param, prior, "data/dr12q/processed/subdla_samples.mat"
    )

    # initialize Bayesian model selection class, with maximum 4 DLAs and at least 1 subDLA,
    # which is the same as Ho-Bird-Garnett (2020).
    bayes = BayesModelSelect([0, 1, max_dlas], 2)  # 0 DLA for null; 1 subDLA; 4 DLAs.
    # 2 for the location of the DLA model in the list

    num_quasars = len(qso_list)

    # allocate the arrays we want to save. Try to save similar variables as in
    # multi_dlas/process_qsos_multiple_dlas_meanflux.m
    # initialize results
    min_z_dlas = np.full((num_quasars,), np.nan)
    max_z_dlas = np.full((num_quasars,), np.nan)
    log_priors_no_dla = np.full((num_quasars,), np.nan)
    log_priors_dla = np.full((num_quasars, max_dlas), np.nan)
    log_likelihoods_no_dla = np.full((num_quasars,), np.nan)
    log_likelihoods_dla = np.full((num_quasars, max_dlas), np.nan)
    log_posteriors_no_dla = np.full((num_quasars,), np.nan)
    log_posteriors_dla = np.full((num_quasars, max_dlas), np.nan)

    sample_log_likelihoods_dla = np.full(
        (num_quasars, param.num_dla_samples, max_dlas), np.nan
    )
    base_sample_inds = np.zeros(
        (num_quasars, param.num_dla_samples, max_dlas - 1), dtype=np.int32
    )

    # initialize lls results
    log_likelihoods_lls = np.full((num_quasars,), np.nan)
    log_posteriors_lls = np.full((num_quasars,), np.nan)
    log_priors_lls = np.full((num_quasars,), np.nan)
    sample_log_likelihoods_lls = np.full((num_quasars, param.num_dla_samples), np.nan)

    # save maps: add the initializations of MAP values
    # N * (1~k models) * (1~k MAP dlas)
    MAP_z_dlas = np.full((num_quasars, max_dlas, max_dlas), np.nan)
    MAP_log_nhis = np.full((num_quasars, max_dlas, max_dlas), np.nan)
    MAP_inds = np.full((num_quasars, max_dlas, max_dlas), np.nan)

    # save model_posteriors in real-scale not in log-scale
    model_posteriors = np.full((num_quasars, 1 + 1 + max_dlas,), np.nan)
    p_dlas = np.full((num_quasars,), np.nan)
    p_no_dlas = np.full((num_quasars,), np.nan)

    for quasar_ind, (filename, z_qso) in enumerate(zip(qso_list, z_qso_list)):
        tic = time.time()

        wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
        rest_wavelengths = param.emitted_wavelengths(wavelengths, z_qso)

        # Null model GP : a GP model without any DLA intervening, we also don't need to
        # run QMC sampling on this model.
        gp = NullGPMAT(
            param,
            prior,
            "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        )

        gp.set_data(
            rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
        )

        # DLA Model GP : this is null model GP + Voigt profile, which is parameterised with
        # {(z_dla, logNHI)}_{i=1}^{k} parameters. k stands for maximum DLA we want to model.
        # we will estimate log posteriors for DLA(1), ..., DLA(k) models.
        dla_gp = DLAGPMAT(
            param,
            prior,
            dla_samples,
            3000.0,
            "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        )
        dla_gp.set_data(
            rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
        )

        subdla_gp = SubDLAGPMAT(
            param,
            prior,
            subdla_samples,
            3000.0,
            "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        )
        subdla_gp.set_data(
            rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
        )

        # run bayesian model selection and get log posteriors
        log_posteriors = bayes.model_selection([gp, subdla_gp, dla_gp], z_qso)
        # at the same time, log_priors, log_likelihoods, log_posteriors will be saved to
        # the bayes instance.

        # prepare to store the values to arrays
        min_z_dlas[quasar_ind] = dla_gp.params.min_z_dla(wavelengths, z_qso)
        max_z_dlas[quasar_ind] = dla_gp.params.max_z_dla(wavelengths, z_qso)

        log_priors_no_dla[quasar_ind] = bayes.log_priors[0]  # null model is index 0
        log_priors_dla[quasar_ind, :] = bayes.log_priors[-max_dlas:]
        log_priors_lls[quasar_ind] = bayes.log_priors[1]  # subDLA model is index 1

        # null model is index 0; subDLA model is index 1
        log_likelihoods_no_dla[quasar_ind] = bayes.log_likelihoods[0]
        log_likelihoods_dla[quasar_ind, :] = bayes.log_likelihoods[-max_dlas:]
        log_likelihoods_lls[quasar_ind] = bayes.log_likelihoods[1]

        # null model is index 0; subDLA model is index 1
        log_posteriors_no_dla[quasar_ind] = bayes.log_posteriors[0]
        log_posteriors_dla[quasar_ind, :] = bayes.log_posteriors[-max_dlas:]
        log_posteriors_lls[quasar_ind] = bayes.log_posteriors[1]

        sample_log_likelihoods_dla[quasar_ind, :, :] = dla_gp.sample_log_likelihoods[
            :, :
        ]
        base_sample_inds[quasar_ind, :, :] = dla_gp.base_sample_inds[:, :].T

        sample_log_likelihoods_lls[quasar_ind, :] = subdla_gp.sample_log_likelihoods[:, 0]

        # save maps: add the initializations of MAP values
        # N * (1~k models) * (1~k MAP dlas)
        MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()

        MAP_z_dlas[quasar_ind, :, :] = MAP_z_dla[:, :]
        MAP_log_nhis[quasar_ind, :, :] = MAP_log_nhi[:, :]

        # save model_posteriors in real-scale not in log-scale
        model_posteriors[quasar_ind, :] = bayes.model_posteriors[:]
        p_dlas[quasar_ind] = bayes.p_dla
        p_no_dlas[quasar_ind] = bayes.p_no_dla

        toc = time.time()
        # very time consuming: ~ 4 mins for a single spectrum without parallelized.
        print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))


    # write into HDF5 file
    with h5py.File("processed_qsos_multi_meanflux.h5", "w") as f:
        # storing default parameter settings for the model
        f.create_dataset(
            "prior_z_qso_increase", data=dla_gp.params.prior_z_qso_increase
        )
        f.create_dataset("k", data=dla_gp.params.k)
        f.create_dataset(
            "normalization_min_lambda", data=dla_gp.params.normalization_min_lambda
        )
        f.create_dataset(
            "normalization_max_lambda", data=dla_gp.params.normalization_max_lambda
        )
        f.create_dataset("min_z_cut", data=dla_gp.params.min_z_cut)
        f.create_dataset("max_z_cut", data=dla_gp.params.max_z_cut)
        f.create_dataset("num_dla_samples", data=dla_gp.params.num_dla_samples)
        f.create_dataset("num_lines", data=dla_gp.params.num_lines)
        f.create_dataset("num_forest_lines", data=dla_gp.params.num_forest_lines)

        # storing the sampling variables
        f.create_dataset("min_z_dlas", data=min_z_dlas)
        f.create_dataset("max_z_dlas", data=max_z_dlas)

        f.create_dataset("sample_log_likelihoods_dla", data=sample_log_likelihoods_dla)
        f.create_dataset("base_sample_inds", data=base_sample_inds)

        f.create_dataset("log_priors_no_dla", data=log_priors_no_dla)
        f.create_dataset("log_priors_lls", data=log_priors_lls)
        f.create_dataset("log_priors_dla", data=log_priors_dla)

        f.create_dataset("log_likelihoods_no_dla", data=log_likelihoods_no_dla)
        f.create_dataset("log_likelihoods_lls", data=log_likelihoods_lls)
        f.create_dataset("log_likelihoods_dla", data=log_likelihoods_dla)

        f.create_dataset("log_posteriors_no_dla", data=log_posteriors_no_dla)
        f.create_dataset("log_posteriors_lls", data=log_posteriors_lls)
        f.create_dataset("log_posteriors_dla", data=log_posteriors_dla)

        f.create_dataset("MAP_z_dlas", data=MAP_z_dlas)
        f.create_dataset("MAP_log_nhis", data=MAP_log_nhis)

        f.create_dataset("p_dlas", data=p_dlas)
        f.create_dataset("p_no_dlas", data=p_no_dlas)
        f.create_dataset("model_posteriors", data=model_posteriors)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--qso_list", nargs="+")
    parser.add_argument("--z_qso_list", nargs="+", type=float)
    parser.add_argument("--max_dlas", type=int, default=4)

    args = parser.parse_args()

    process_qso(
        args.qso_list, args.z_qso_list, read_spec.read_spec, max_dlas=args.max_dlas
    )
